"""
jepa_inference.py  —  JEPA Inference with Iterative Self-Refinement
====================================================================
Fully self-contained. No imports from jepa_train.py needed.

What changed from v2
--------------------
• PatchContextEncoder added (required by new checkpoint).
• execute_inference now accepts `num_refinement_passes` (default 2).
  Each pass feeds the previous prediction back as patch context,
  allowing the model to iteratively denoise its own visual tokens.
  This is in-distribution at inference because the model was trained
  on exactly this: pass-1 predictions as context for pass-2.

  Pass 0 (text-only)  : baseline, no GT or predicted patches
  Pass 1 (refinement) : pass-0 predictions as patch context
  Pass N (refinement) : pass-(N-1) predictions as patch context

  The CSV saves the final-pass prediction. You can set
  num_refinement_passes=0 for pure text-only (v2 behaviour),
  or 1-3 for iterative refinement.

Checkpoint format (saved by jepa_train.py v3):
  {
      "text_encoder":  OrderedDict,
      "patch_encoder": OrderedDict,
      "predictor":     OrderedDict,
      "config": { "text_dim", "hidden_dim", "num_queries", ... }
  }
"""

import os
import torch
import torch.nn as nn
import csv
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import load_dataset
from tqdm import tqdm

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture — must match jepa_train.py exactly
# ══════════════════════════════════════════════════════════════════════════════

class TextContextEncoder(nn.Module):
    def __init__(
        self,
        text_dim:   int   = 4096,
        hidden_dim: int   = 1024,
        num_heads:  int   = 16,
        num_layers: int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(text_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_embeddings, attention_mask=None):
        x   = self.input_proj(text_embeddings)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        return self.norm(self.transformer(x, src_key_padding_mask=kpm))


class PatchContextEncoder(nn.Module):
    """
    Adds learnable positional embeddings to patch tokens.
    At inference, patch_indices = all 576 positions (full grid).
    """
    def __init__(self, hidden_dim: int = 1024, num_patches: int = 576):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patches, patch_indices):
        pos = self.pos_embed[:, patch_indices, :]
        return self.norm(patches + pos)


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim:  int = 1024,
        num_queries: int = 576,
        num_heads:   int = 8,
        num_layers:  int = 4,
    ):
        super().__init__()
        self.pos_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=0.0, batch_first=True, norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, context, mask_indices=None):
        B       = context.shape[0]
        queries = self.pos_queries.expand(B, -1, -1)
        if mask_indices is not None:
            queries = queries[:, mask_indices, :]
        return self.norm(self.transformer(queries, context))


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint loader
# ══════════════════════════════════════════════════════════════════════════════

def load_jepa_models(checkpoint_path: str, device: torch.device):
    """
    Load all three JEPA models from checkpoint.
    Returns (text_encoder, patch_encoder, predictor) — all eval on `device`.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    required = {"text_encoder", "patch_encoder", "predictor", "config"}
    missing  = required - set(ckpt.keys())
    if missing:
        raise KeyError(
            f"Checkpoint missing keys: {missing}\n"
            f"Found: {list(ckpt.keys())}\n"
            f"This inference file requires a v3 checkpoint (jepa_train.py v3).\n"
            f"Re-run training to generate a compatible jepa_crossmodal.pth."
        )

    cfg = ckpt["config"]
    print(f"Checkpoint config: {cfg}")

    text_encoder = TextContextEncoder(
        text_dim   = cfg.get("text_dim",   4096),
        hidden_dim = cfg.get("hidden_dim", 1024),
        num_heads  = 16, num_layers = 4,
    ).to(device)
    text_encoder.load_state_dict(ckpt["text_encoder"])
    text_encoder.eval()

    patch_encoder = PatchContextEncoder(
        hidden_dim  = cfg.get("hidden_dim",  1024),
        num_patches = cfg.get("num_queries", 576),
    ).to(device)
    patch_encoder.load_state_dict(ckpt["patch_encoder"])
    patch_encoder.eval()

    predictor = JEPAPredictor(
        hidden_dim  = cfg.get("hidden_dim",  1024),
        num_queries = cfg.get("num_queries", 576),
        num_heads   = 8, num_layers = 4,
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    return text_encoder, patch_encoder, predictor


# ══════════════════════════════════════════════════════════════════════════════
#  Inference pipeline
# ══════════════════════════════════════════════════════════════════════════════

def execute_inference(
    checkpoint_path:       str = "jepa_crossmodal.pth",
    num_samples:           int = 100,
    hint_min_len:          int = 35,
    max_text_len:          int = 128,
    max_new_tokens:        int = 128,
    num_refinement_passes: int = 2,     # 0 = text-only; 1-3 = iterative refinement
    output_csv:            str = "jepa_llava_results.csv",
):
    """
    Args
    ----
    num_refinement_passes : int
        0  →  pure text-only prediction (same as v2, no refinement)
        1  →  one refinement: text → pred_0 → (text + pred_0 as context) → pred_1
        2  →  two refinements (recommended; diminishing returns beyond 3)
        Higher values increase GPU memory and latency proportionally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Refinement passes: {num_refinement_passes}")

    # ── LLaVA ─────────────────────────────────────────────────────────────
    print("Loading LLaVA-1.5-7B...")
    processor   = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    llava_model = (
        LlavaForConditionalGeneration
        .from_pretrained("llava-hf/llava-1.5-7b-hf")
        .to(device).eval()
    )

    # ── Vicuna ────────────────────────────────────────────────────────────
    print("Loading Vicuna-7B embedding layer...")
    vicuna_tok = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
    if vicuna_tok.pad_token is None:
        vicuna_tok.pad_token = vicuna_tok.eos_token
    vicuna_llm   = (
        AutoModelForCausalLM
        .from_pretrained("lmsys/vicuna-7b-v1.5")
        .to(device).eval()
    )
    vicuna_embed = vicuna_llm.get_input_embeddings()

    # ── JEPA ──────────────────────────────────────────────────────────────
    print(f"Loading JEPA checkpoint: {checkpoint_path}")
    text_encoder, patch_encoder, predictor = load_jepa_models(checkpoint_path, device)

    # ── ScienceQA ─────────────────────────────────────────────────────────
    print("Loading ScienceQA...")
    dataset  = load_dataset("derek-thomas/ScienceQA", split="train")
    filtered = [
        row for row in dataset
        if isinstance(row.get("hint", ""), str) and len(row["hint"]) > hint_min_len
    ][:num_samples]
    print(f"Using {len(filtered)} samples (hint length > {hint_min_len})")

    all_idx      = torch.arange(576, device=device)   # all patch positions
    rows_to_write = []

    for row in tqdm(filtered, desc="JEPA-LLaVA inference"):
        hint         = row.get("hint", "")
        question     = row.get("question", "")
        choices      = row.get("choices", [])
        answer_idx   = row.get("answer", 0)
        correct_opt  = choices[answer_idx]
        options_text = "\n".join(f"{i}. {c}" for i, c in enumerate(choices))
        caption      = hint if hint else question

        # ── Step 1: Vicuna embed ───────────────────────────────────────────
        tok_out = vicuna_tok(
            caption, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_text_len,
        ).to(device)

        with torch.no_grad():
            raw_text_emb = vicuna_embed(tok_out.input_ids)             # [1, T, 4096]

        # ── Step 2: Text context ───────────────────────────────────────────
        with torch.no_grad():
            text_ctx = text_encoder(raw_text_emb, tok_out.attention_mask)  # [1, T, 1024]

        # ── Step 3: Iterative self-refinement ─────────────────────────────
        #
        # Pass 0 (always): predict from text context alone.
        #   This is the "cold start" — no GT image, no prior predictions.
        #
        # Pass k (k ≥ 1): concatenate text context with position-encoded
        #   predictions from pass k-1, then predict again.
        #   The model was trained on exactly this loop, so it is
        #   in-distribution for all passes up to num_refinement_passes.
        #
        # Intuition: each pass refines spatial structure and semantic
        # alignment, similar to how diffusion models denoise over steps.

        with torch.no_grad():
            # Pass 0: text only
            z_img = predictor(text_ctx)                                # [1, 576, 1024]

            # Refinement passes
            for _pass in range(num_refinement_passes):
                # Encode previous predictions as patch context
                patch_ctx    = patch_encoder(z_img, all_idx)          # [1, 576, 1024]
                full_ctx     = torch.cat([text_ctx, patch_ctx], dim=1) # [1, T+576, 1024]
                z_img        = predictor(full_ctx)                     # [1, 576, 1024]

        # ── Step 4: Project to LLaVA visual space ─────────────────────────
        with torch.no_grad():
            h_vis = llava_model.multi_modal_projector(z_img)           # [1, 576, llava_D]

        # ── Step 5: LLaVA generation ───────────────────────────────────────
        llava_prompt = (
            f"USER: <image>\n"
            f"Context: {hint}\n"
            f"Question: {question}\n"
            f"Options:\n{options_text}\n"
            f"Select the correct option.\n"
            f"ASSISTANT:"
        )
        inputs    = processor(text=llava_prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        img_tok_id  = llava_model.config.image_token_index
        img_tok_pos = (input_ids == img_tok_id).nonzero(as_tuple=True)[1][0]

        with torch.no_grad():
            text_emb = llava_model.get_input_embeddings()(input_ids)
            prefix   = text_emb[:, :img_tok_pos, :]
            suffix   = text_emb[:, img_tok_pos + 1:, :]
            inp_emb  = torch.cat([prefix, h_vis, suffix], dim=1)
            attn     = torch.ones(inp_emb.shape[:2], dtype=torch.long, device=device)

            out_ids = llava_model.generate(
                inputs_embeds  = inp_emb,
                attention_mask = attn,
                max_new_tokens = max_new_tokens,
                use_cache      = True,
            )

        generated  = processor.tokenizer.decode(
            out_ids[0], skip_special_tokens=True
        ).strip()
        prediction = generated.split("ASSISTANT:")[-1].strip()

        rows_to_write.append([
            question, hint, " | ".join(choices), correct_opt, prediction,
        ])

    # ── Save CSV ───────────────────────────────────────────────────────────
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["question", "hint", "options", "correct_answer", "model_prediction"]
        )
        writer.writerows(rows_to_write)

    print(f"Done. Saved → {output_csv}  ({len(rows_to_write)} samples)")


if __name__ == "__main__":
    execute_inference()
