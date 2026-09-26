"""
jepa_train_medical_v2.py  —  Cross-Modal JEPA Training  (Medical, PMC-VQA)
===========================================================================

KEY ARCHITECTURAL FIXES OVER v1
─────────────────────────────────
FIX 1 ── CLIP model changed to openai/clip-vit-large-patch14-336
         This is the EXACT same CLIP that LLaVA-1.5 was trained with.
         Consequence: the visual tokens our JEPA predictor learns to
         generate live in the same feature space that LLaVA's
         multi_modal_projector was trained to consume.
         Previously (BiomedCLIP ViT-B/16) there was a distribution
         mismatch — correct dimension (1024) but completely wrong
         feature distribution.

         NUM_VIS_TOKENS : 196  →  576   (24×24 grid, patch-14, 336px)
         CLIP_DIM       : 768  →  1024  (ViT-L hidden size)

FIX 2 ── PatchContextEncoder.input_proj removed
         CLIP-L outputs 1024-dim = HIDDEN_DIM = 1024.
         The 768→1024 projection was only needed because BiomedCLIP
         produced 768-dim tokens. Now dims match natively — no projection,
         no information distortion, just positional embedding addition.

FIX 3 ── LossHead removed entirely
         Previously: predictor (1024) → LossHead (1024→768) → compare
         with BiomedCLIP targets (768). Needed because CLIP dim ≠ JEPA dim.
         Now: predictor (1024) → compare directly with CLIP-L targets (1024).
         Fewer parameters, cleaner gradient flow, no intermediate distortion.

FIX 4 ── Self-refinement loop simplified
         Previously loss_head(pred_1024) → 768 → PatchContextEncoder(768→1024).
         Now pred_1024 → PatchContextEncoder directly (no projections at all).
         The refinement feedback stays in the same 1024-dim space throughout.

WHAT STAYS THE SAME
────────────────────
  TextContextEncoder   : unchanged (4096→1024 projection still needed)
  JEPAPredictor        : unchanged architecture, just no out_proj
  Dynamic masking      : Uniform(0.50, 1.00) each step
  Patch annealing      : cosine 0.50 → 0.0 over first 80% of training
  Self-refinement      : 2-pass loop, λ = 0.5
  Loss function        : Smooth-L1 on L2-normalised vectors
  Optimiser            : AdamW + cosine LR + warmup
  Dataset              : PMC-VQA  (train_2.csv)
  Text encoder         : PMC-LLaMA 7B  (frozen embedding layer only)

DATASET SPLITS
──────────────
  TRAINING   : train_2.csv  —  rows after first val_size (shuffled)
  VALIDATION : train_2.csv  —  first val_size rows (shuffled, seed=42)
  TEST       : test_2.csv   —  completely held-out, NEVER seen here
               (test_2.csv is only used in the inference scripts)

CHECKPOINT
──────────
  Saved to : /scratch/knishant_iitp/jepa_medical_v2.pth
  Keys     : text_encoder | patch_encoder | predictor | config
             (no loss_head — it no longer exists)
"""

import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from PIL import Image

# ── HuggingFace cache — must be set before any HF import fires ────────────────
HF_CACHE_DIR = "/scratch/knishant_iitp/"
os.environ["HF_HOME"]               = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]    = HF_CACHE_DIR

from transformers import (
    CLIPImageProcessor,   # standard HF processor for CLIP-L
    CLIPVisionModel,      # standard HF vision model for CLIP-L
    AutoTokenizer,
    AutoModelForCausalLM,
)
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
#  All dimension constants flow from the CLIP model choice.
#  With CLIP-ViT-L/14@336:  CLIP_DIM = HIDDEN_DIM = 1024  (no mismatch)
# ══════════════════════════════════════════════════════════════════════════════

NUM_VIS_TOKENS = 576    # CLIP-ViT-L/14 @ 336px → 24×24 patches = 576 tokens
CLIP_DIM       = 1024   # ViT-L hidden size  (= HIDDEN_DIM — no projection needed)
TEXT_DIM       = 4096   # PMC-LLaMA 7B LLaMA-family embedding dimension
HIDDEN_DIM     = 1024   # JEPA internal dimension throughout

# Paths
IMAGE_ROOT = "./PMC-VQA/figures/"
TRAIN_CSV  = "./PMC-VQA/train_2.csv"
# NOTE: test_2.csv is NEVER loaded here — it is reserved for inference only

# Model IDs
CLIP_ID      = "openai/clip-vit-large-patch14-336"   # same CLIP as LLaVA-1.5
PMC_LLAMA_ID = "chaoyi-wu/PMC_LLaMA_7B"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class TextContextEncoder(nn.Module):
    """
    Encodes PMC-LLaMA token embeddings into JEPA context vectors.

    Why this projection exists:
      PMC-LLaMA 7B embedding dim = 4096.
      JEPA hidden dim = 1024.
      This Linear(4096→1024) is the only unavoidable projection —
      we must down-project LLaMA embeddings to the working dim.

    Architecture: Pre-LN TransformerEncoder (wide FFN × 4)
      input_proj  : Linear(4096 → 1024)
      4 × encoder layers, 16 attention heads
      FFN hidden  : 1024 × 4 = 4096
      dropout     : 0.1
    """

    def __init__(
        self,
        text_dim:   int   = TEXT_DIM,    # 4096
        hidden_dim: int   = HIDDEN_DIM,  # 1024
        num_heads:  int   = 16,
        num_layers: int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(text_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model        = hidden_dim,
                nhead          = num_heads,
                dim_feedforward= hidden_dim * 4,   # wide FFN
                dropout        = dropout,
                batch_first    = True,
                norm_first     = True,             # Pre-LN for stability
            ),
            num_layers = num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,     # [B, T, 4096]
        attention_mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:                     # [B, T, 1024]
        x   = self.input_proj(text_embeddings)
        # Convert HF attention mask (1=attend, 0=pad) to PyTorch
        # key_padding_mask convention (True=ignore)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        return self.norm(self.transformer(x, src_key_padding_mask=kpm))


class PatchContextEncoder(nn.Module):
    """
    Adds learnable spatial positional embeddings to CLIP patch tokens.

    Why NO input_proj here (unlike v1):
      CLIP-ViT-L/14 outputs 1024-dim tokens.
      JEPA hidden dim is also 1024.
      Dims already match — adding a projection would only introduce
      unnecessary parameters and potential distortion.

    The only operation is: patches + pos_embed → LayerNorm.

    Used in two contexts:
      (a) Training scaffolding phase  : GT CLIP patches (visible subset)
      (b) Self-refinement             : predictor output from previous pass
          Both are 1024-dim, so the same encoder handles both cleanly.

    num_patches = 576  (full 24×24 spatial grid from CLIP-L)
    """

    def __init__(
        self,
        hidden_dim:  int = HIDDEN_DIM,      # 1024
        num_patches: int = NUM_VIS_TOKENS,  # 576
    ):
        super().__init__()
        # Learnable position table covering all 576 patch positions
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        patches:       torch.Tensor,   # [B, V, 1024]  V ≤ 576
        patch_indices: torch.Tensor,   # [V]  which positions in 0..575
    ) -> torch.Tensor:                 # [B, V, 1024]
        # Slice positional embeddings for the given spatial positions
        pos = self.pos_embed[:, patch_indices, :]   # [1, V, 1024]
        return self.norm(patches + pos)             # [B, V, 1024]


class JEPAPredictor(nn.Module):
    """
    Cross-attention Transformer decoder that predicts masked CLIP visual tokens
    from the combined text + visible-patch context.

    Why NO out_proj here (unlike v1):
      Previously: out_proj projected 1024 → 768 to match BiomedCLIP targets.
      Now: CLIP-L targets are 1024-dim = predictor output dim.
      Predictor output is directly comparable to CLIP targets for the loss
      AND directly compatible with LLaVA's multi_modal_projector input.
      One module, two uses, zero extra projections.

    Architecture: Pre-LN TransformerDecoder (narrow FFN × 2)
      Learnable position queries : [1, 576, 1024]
      4 × decoder layers, 8 attention heads
      FFN hidden : 1024 × 2 = 2048  (narrow — predictor should be leaner)
      No dropout during training (common in JEPA predictors)

    At inference: mask_indices=None → predicts all 576 tokens.
    """

    def __init__(
        self,
        hidden_dim:  int = HIDDEN_DIM,      # 1024
        num_queries: int = NUM_VIS_TOKENS,  # 576
        num_heads:   int = 8,
        num_layers:  int = 4,
    ):
        super().__init__()
        # One learnable query per patch position (spatial prior)
        self.pos_queries = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim)
        )
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model        = hidden_dim,
                nhead          = num_heads,
                dim_feedforward= hidden_dim * 2,   # narrow FFN
                dropout        = 0.0,
                batch_first    = True,
                norm_first     = True,             # Pre-LN
            ),
            num_layers = num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        context:      torch.Tensor,              # [B, T+V, 1024]  full memory
        mask_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # [B, M, 1024]
        B       = context.shape[0]
        queries = self.pos_queries.expand(B, -1, -1)   # [B, 576, 1024]
        if mask_indices is not None:
            # Only activate queries for masked positions
            queries = queries[:, mask_indices, :]       # [B, M, 1024]
        return self.norm(self.transformer(queries, context))


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING MECHANISM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_visible_ratio(
    step:            int,
    total_steps:     int,
    initial_ratio:   float = 0.50,
    anneal_fraction: float = 0.80,
) -> float:
    """
    Patch annealing schedule — cosine decay of GT visible patch fraction.

    Phase 1 (steps 0 → anneal_steps):
      Cosine decay from initial_ratio (0.50) → 0.0
      The predictor gets GT patches as scaffolding, helping it learn
      spatial structure before being forced to predict from text alone.

    Phase 2 (steps anneal_steps → total_steps):
      vis_ratio locked at 0.0 — pure text-to-vision prediction.
      This consolidation phase ensures the model is fully in the
      inference regime (which has no GT patches) before training ends.

    Returns: visible_ratio in [0.0, initial_ratio]
    """
    anneal_steps = int(total_steps * anneal_fraction)
    if step >= anneal_steps:
        return 0.0
    progress = step / anneal_steps
    return initial_ratio * 0.5 * (1.0 + math.cos(math.pi * progress))


def sample_mask(
    num_tokens:     int,
    device:         torch.device,
    mask_ratio_min: float = 0.50,
    mask_ratio_max: float = 1.00,
) -> tuple[torch.Tensor, float]:
    """
    Dynamic masking — sample mask_ratio ~ Uniform(min, max) each step.

    Why dynamic (not fixed):
      Fixed ratio (e.g. 0.75) never exposes the model to the inference
      regime where ALL tokens are masked (ratio=1.0).
      Uniform sampling trains the model across the full spectrum,
      including ratio=1.0 which IS inference.

    Returns: (sorted masked indices [M], sampled mask_ratio)
    """
    mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)
    num_masked = min(num_tokens, max(1, int(num_tokens * mask_ratio)))
    perm       = torch.randperm(num_tokens, device=device)
    return perm[:num_masked].sort().values, mask_ratio


def jepa_loss(
    predicted: torch.Tensor,  # [B, M, 1024]
    target:    torch.Tensor,  # [B, M, 1024]
) -> torch.Tensor:
    """
    JEPA loss: Smooth-L1 between L2-normalised predicted and target tokens.

    L2 normalisation before loss:
      Forces the model to learn DIRECTIONS in feature space rather than
      magnitudes. This prevents collapse to zero-vector predictions and
      is standard practice in JEPA / self-supervised learning.

    Stop-gradient on target:
      Target CLIP tokens must be detached so gradients do not flow
      back into the frozen CLIP encoder. This is enforced by .detach().
    """
    return F.smooth_l1_loss(
        F.normalize(predicted,       dim=-1),
        F.normalize(target.detach(), dim=-1),
    )


def build_full_context(
    text_ctx:      torch.Tensor,          # [B, T, 1024]
    patches:       torch.Tensor | None,   # [B, V, 1024] or None
    patch_indices: torch.Tensor | None,   # [V] or None
    patch_encoder: PatchContextEncoder,
) -> torch.Tensor:                        # [B, T+V, 1024] or [B, T, 1024]
    """
    Concatenate text context with position-encoded patch context.

    patches=None happens when:
      (a) vis_ratio has annealed to 0.0 (no GT patches exposed)
      (b) First refinement pass before any patches are predicted

    In these cases, text context alone is returned unchanged.
    """
    if patches is None or patch_indices is None or patches.shape[1] == 0:
        return text_ctx
    patch_ctx = patch_encoder(patches, patch_indices)          # [B, V, 1024]
    return torch.cat([text_ctx, patch_ctx], dim=1)             # [B, T+V, 1024]


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_caption(row: pd.Series) -> str:
    """
    Use the Caption column (figure caption from the PMC paper) as the
    text input to JEPA. This is correct because:
      - The caption DESCRIBES the image content
      - JEPA's task is to predict visual tokens FROM text
      - A description is the right grounding signal for that task

    Fallback to Question text if Caption is missing (rare edge case).

    NOTE: The Question + Choices are used in the inference prompt for
    LLaVA — they are NOT used as JEPA text input here.
    """
    caption = str(row.get("Caption", "")).strip()
    return caption if caption else str(row.get("Question", "")).strip()


def load_image_safe(path: str) -> Image.Image | None:
    """Load PIL image, return None on any file error (missing/corrupt)."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def prepare_dataframe(
    csv_path: str,
    val_size: int,
    seed:     int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train_2.csv, shuffle, split into train and validation sets.

    IMPORTANT — what data is used where:
      This function only loads TRAIN_CSV (train_2.csv).
      test_2.csv is NEVER touched here.

      train_2.csv  →  shuffle  →  first val_size rows  =  VALIDATION
                                  remaining rows         =  TRAINING

      The validation set is pre-computed once before training starts
      and used to track JEPA loss across checkpoints. It does NOT
      contain any test_2.csv samples — there is no data leakage.

    Returns: (train_df, val_df)
    """
    required = [
        "Figure_path", "Question", "Answer", "Caption",
        "Choice A", "Choice B", "Choice C", "Choice D",
    ]
    df = (
        pd.read_csv(csv_path)
          .dropna(subset=required)
          .sample(frac=1, random_state=seed)
          .reset_index(drop=True)
    )
    val_df   = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    print(f"  train_2.csv split → train: {len(train_df):,}  |  val: {len(val_df):,}")
    print(f"  test_2.csv is held-out — not loaded here")
    return train_df, val_df


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def execute_training_pipeline(
    total_steps:       int   = 20_000,
    # ── Patch annealing ──────────────────────────────────────────────────
    initial_vis_ratio: float = 0.50,   # GT patch fraction at step 0
    anneal_fraction:   float = 0.80,   # decay over first 80% of training
    # ── Dynamic masking ──────────────────────────────────────────────────
    mask_ratio_min:    float = 0.50,
    mask_ratio_max:    float = 1.00,
    # ── Self-refinement ──────────────────────────────────────────────────
    lambda_refine:     float = 0.50,   # weight of refinement loss
    # ── Optimisation ─────────────────────────────────────────────────────
    val_size:          int   = 300,
    val_every:         int   = 200,
    lr:                float = 1e-4,
    weight_decay:      float = 0.05,
    max_text_len:      int   = 128,
):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  JEPA Medical Training  v2  (CLIP-L/14@336)")
    print("=" * 65)
    print(f"  Device             : {device}")
    print(f"  CLIP model         : {CLIP_ID}")
    print(f"  NUM_VIS_TOKENS     : {NUM_VIS_TOKENS}  (24×24, no projection needed)")
    print(f"  CLIP_DIM           : {CLIP_DIM}  (= HIDDEN_DIM, clean alignment)")
    print(f"  Text model         : {PMC_LLAMA_ID}")
    print(f"  Total steps        : {total_steps}")
    print(f"  Patch annealing    : {initial_vis_ratio} → 0.0 over "
          f"first {anneal_fraction*100:.0f}% of training")
    print(f"  Mask ratio         : Uniform({mask_ratio_min}, {mask_ratio_max})")
    print(f"  Refinement λ       : {lambda_refine}")
    print("=" * 65)

    # ── 1. FROZEN CLIP  (target encoder — provides GT visual tokens) ──────
    # Using CLIP-ViT-L/14@336 — the SAME model embedded inside LLaVA-1.5.
    # This guarantees our predicted tokens are in the correct distribution
    # for LLaVA's multi_modal_projector at inference time.
    print(f"\nLoading CLIP target encoder: {CLIP_ID} ...")
    clip_processor = CLIPImageProcessor.from_pretrained(
        CLIP_ID, cache_dir=HF_CACHE_DIR
    )
    clip_model = CLIPVisionModel.from_pretrained(
        CLIP_ID, cache_dir=HF_CACHE_DIR
    ).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)   # CLIP is frozen — only provides targets

    # ── 2. FROZEN PMC-LLaMA EMBEDDING LAYER  (text input encoder) ─────────
    # We only need the embedding lookup table (token ID → 4096-dim vector).
    # The full 7B model is loaded to CPU in float16, embedding extracted,
    # then the rest is deleted to free memory before GPU training starts.
    print(f"Loading PMC-LLaMA embedding layer: {PMC_LLAMA_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        PMC_LLAMA_ID, use_fast=False, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        PMC_LLAMA_ID,
        torch_dtype = torch.float16,
        device_map  = "cpu",          # load to CPU first
        cache_dir   = HF_CACHE_DIR,
    )
    text_embed_layer = llm.get_input_embeddings().to(device)
    text_embed_layer.eval()
    for p in text_embed_layer.parameters():
        p.requires_grad_(False)       # embedding table is frozen

    del llm                           # free the 7B weights — not needed
    torch.cuda.empty_cache()

    # ── 3. LEARNABLE JEPA COMPONENTS ──────────────────────────────────────
    # Three modules are trained; all others are frozen.
    # No LossHead — predictor outputs 1024-dim directly matching CLIP-L.
    text_encoder  = TextContextEncoder().to(device)
    patch_encoder = PatchContextEncoder().to(device)
    predictor     = JEPAPredictor().to(device)

    all_params = (
        list(text_encoder.parameters())  +
        list(patch_encoder.parameters()) +
        list(predictor.parameters())
    )
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in all_params) / 1e6:.1f}M")

    # ── 4. OPTIMISER + LR SCHEDULE ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        all_params, lr=lr, weight_decay=weight_decay
    )

    def lr_lambda(step):
        """Linear warmup → cosine decay."""
        warmup = 400
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 5. DATASET LOADING AND SPLIT ──────────────────────────────────────
    print(f"\nLoading training data from {TRAIN_CSV} ...")
    train_df, val_df = prepare_dataframe(TRAIN_CSV, val_size)
    train_cursor = 0   # pointer into train_df, cycles when exhausted

    # ── 6. PRE-COMPUTE VALIDATION SET ─────────────────────────────────────
    # All val CLIP tokens and text embeddings are extracted once and cached
    # on CPU. This avoids re-running CLIP and PMC-LLaMA at every val step.
    print(f"Pre-computing {len(val_df)} validation samples ...")
    val_data = []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img_path = os.path.join(IMAGE_ROOT, row["Figure_path"])
            image    = load_image_safe(img_path)
            if image is None:
                continue

            # CLIP-L patch tokens  [1, 576, 1024]
            pv = clip_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device)
            clip_out    = clip_model(pv, output_hidden_states=True)
            clip_tokens = clip_out.hidden_states[-2][:, 1:, :]  # [1, 576, 1024]

            # PMC-LLaMA text embeddings  [1, T, 4096]
            caption = build_caption(row)
            tok = tokenizer(
                caption, return_tensors="pt",
                truncation=True, max_length=max_text_len,
                padding="max_length",
            ).to(device)
            raw_text_emb = text_embed_layer(tok.input_ids).float()

            val_data.append({
                "clip_tokens":    clip_tokens.cpu(),
                "raw_text_emb":   raw_text_emb.cpu(),
                "attention_mask": tok.attention_mask.cpu(),
            })

    print(f"Validation set ready: {len(val_data)} samples")

    # ── 7. TRACKING VARIABLES ─────────────────────────────────────────────
    train_losses   = []
    val_losses_p1  = []   # pass-1 val loss (text-only prediction)
    val_losses_p2  = []   # pass-2 val loss (after 1 refinement)
    val_steps      = []
    mask_ratio_log = []
    vis_ratio_log  = []

    # ── 8. MAIN TRAINING LOOP ─────────────────────────────────────────────
    print(f"\nStarting training for {total_steps} steps ...\n")

    for step in range(total_steps):
        text_encoder.train()
        patch_encoder.train()
        predictor.train()

        # Cycle through training data
        row = train_df.iloc[train_cursor % len(train_df)]
        train_cursor += 1

        img_path = os.path.join(IMAGE_ROOT, row["Figure_path"])
        image    = load_image_safe(img_path)
        if image is None:
            # Skip corrupt/missing images without breaking the step count
            train_losses.append(train_losses[-1] if train_losses else 0.0)
            mask_ratio_log.append(0.75)
            vis_ratio_log.append(get_visible_ratio(step, total_steps))
            scheduler.step()
            continue

        # ── Extract frozen features (no gradient) ─────────────────────────
        with torch.no_grad():
            # CLIP-L patch tokens  [1, 576, 1024]  — training targets
            pv = clip_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device)
            clip_out    = clip_model(pv, output_hidden_states=True)
            clip_tokens = clip_out.hidden_states[-2][:, 1:, :]  # [1, 576, 1024]

            # PMC-LLaMA embeddings  [1, T, 4096]
            caption      = build_caption(row)
            tok = tokenizer(
                caption, return_tensors="pt",
                truncation=True, max_length=max_text_len,
                padding="max_length",
            ).to(device)
            raw_text_emb = text_embed_layer(tok.input_ids).float()

        # ── Step A: Encode text → JEPA context ────────────────────────────
        text_ctx = text_encoder(raw_text_emb, tok.attention_mask)  # [1, T, 1024]

        # ── Step B: Patch annealing — determine visible GT patches ─────────
        # Early training: up to 50% of GT patches given as context (scaffolding)
        # Later training: fraction decays to 0 (pure text-to-vision prediction)
        vis_ratio   = get_visible_ratio(
            step, total_steps, initial_vis_ratio, anneal_fraction
        )
        num_visible = int(NUM_VIS_TOKENS * vis_ratio)
        vis_ratio_log.append(vis_ratio)

        if num_visible > 0:
            # Split patch positions into visible (scaffolding) and maskable
            perm        = torch.randperm(NUM_VIS_TOKENS, device=device)
            visible_idx = perm[:num_visible].sort().values   # [V]  shown to predictor
            mask_pool   = perm[num_visible:].sort().values   # [576-V]  candidates to mask
        else:
            # Fully annealed — all positions are masking candidates
            visible_idx = None
            mask_pool   = torch.arange(NUM_VIS_TOKENS, device=device)

        # ── Step C: Dynamic masking — choose which positions to predict ────
        mask_ratio  = random.uniform(mask_ratio_min, mask_ratio_max)
        num_masked  = min(len(mask_pool), max(1, int(len(mask_pool) * mask_ratio)))
        sub_perm    = torch.randperm(len(mask_pool), device=device)
        masked_idx  = mask_pool[sub_perm[:num_masked]].sort().values  # [M]
        mask_ratio_log.append(mask_ratio)

        # ── Step D: Build context (text + optional GT scaffolding) ─────────
        vis_patches = (
            clip_tokens[:, visible_idx, :]   # [1, V, 1024] — GT patches
            if visible_idx is not None else None
        )
        # PatchContextEncoder adds positional embeddings (no projection needed)
        full_ctx = build_full_context(
            text_ctx, vis_patches, visible_idx, patch_encoder
        )   # [1, T+V, 1024]

        # ── Step E: Primary prediction (JEPA pass 1) ──────────────────────
        # Predictor cross-attends to full context, outputs at masked positions
        predicted  = predictor(full_ctx, mask_indices=masked_idx)  # [1, M, 1024]
        target     = clip_tokens[:, masked_idx, :]                 # [1, M, 1024]
        loss_main  = jepa_loss(predicted, target)

        # ── Step F: Self-refinement (JEPA pass 2) ─────────────────────────
        # Pass 1 (detached): predict ALL 576 tokens from TEXT-ONLY context.
        # This simulates inference starting point — no GT, no scaffolding.
        # Pass 2 (trained): feed pass-1 predictions as "predicted patch context",
        # re-predict the same masked positions. Loss trains the model to
        # improve on its own initial guess — iterative latent refinement.
        #
        # SIMPLIFICATION vs v1:
        #   Previously: pass-1 needed loss_head (1024→768) before
        #               PatchContextEncoder (768→1024) — two projections.
        #   Now: pass-1 output (1024) feeds PatchContextEncoder directly.
        #        Zero intermediate projections — cleaner gradient path.
        with torch.no_grad():
            pred_pass1 = predictor(text_ctx)   # [1, 576, 1024]  text-only
            all_idx    = torch.arange(NUM_VIS_TOKENS, device=device)

        # Encode pass-1 predictions as patch context for pass-2
        refine_ctx  = build_full_context(
            text_ctx, pred_pass1.detach(), all_idx, patch_encoder
        )   # [1, T+576, 1024]
        pred_pass2  = predictor(refine_ctx, mask_indices=masked_idx)  # [1, M, 1024]
        loss_refine = jepa_loss(pred_pass2, target)

        # ── Step G: Total loss and backprop ───────────────────────────────
        loss = loss_main + lambda_refine * loss_refine

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        # ── VALIDATION ────────────────────────────────────────────────────
        if (step + 1) % val_every == 0:
            text_encoder.eval()
            patch_encoder.eval()
            predictor.eval()

            total_vl_p1 = 0.0
            total_vl_p2 = 0.0
            all_idx_val = torch.arange(NUM_VIS_TOKENS, device=device)

            with torch.no_grad():
                for vd in val_data:
                    v_clip = vd["clip_tokens"].to(device)    # [1, 576, 1024]
                    v_text = vd["raw_text_emb"].to(device)
                    v_mask = vd["attention_mask"].to(device)

                    v_text_ctx = text_encoder(v_text, v_mask)

                    # Val pass 1: text-only → all 576 predictions
                    v_pred_p1  = predictor(v_text_ctx)       # [1, 576, 1024]
                    total_vl_p1 += jepa_loss(v_pred_p1, v_clip).item()

                    # Val pass 2: refinement from pass-1 predictions
                    v_refine_ctx = build_full_context(
                        v_text_ctx, v_pred_p1, all_idx_val, patch_encoder
                    )
                    v_pred_p2   = predictor(v_refine_ctx)    # [1, 576, 1024]
                    total_vl_p2 += jepa_loss(v_pred_p2, v_clip).item()

            avg_vl_p1 = total_vl_p1 / len(val_data)
            avg_vl_p2 = total_vl_p2 / len(val_data)
            curr_vis  = get_visible_ratio(
                step, total_steps, initial_vis_ratio, anneal_fraction
            )
            curr_lr   = scheduler.get_last_lr()[0]
            avg_msk   = sum(mask_ratio_log[-val_every:]) / val_every

            val_losses_p1.append(avg_vl_p1)
            val_losses_p2.append(avg_vl_p2)
            val_steps.append(step + 1)

            print(
                f"Step {step+1:>6}/{total_steps} | "
                f"Train: {loss.item():.5f} "
                f"(main={loss_main.item():.5f} ref={loss_refine.item():.5f}) | "
                f"Val P1: {avg_vl_p1:.5f} | Val P2: {avg_vl_p2:.5f} | "
                f"vis: {curr_vis:.3f} | mask: {avg_msk:.3f} | "
                f"LR: {curr_lr:.2e}"
            )

            # 3-panel training plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            ax = axes[0]
            ax.plot(range(1, step + 2), train_losses,
                    alpha=0.3, color="steelblue", label="Train Loss")
            ax.plot(val_steps, val_losses_p1,
                    color="red", linewidth=2, label="Val P1 (text-only)")
            ax.plot(val_steps, val_losses_p2,
                    color="green", linewidth=2, linestyle="--",
                    label="Val P2 (1 refinement)")
            ax.set_xlabel("Step"); ax.set_ylabel("Smooth-L1 (normalised)")
            ax.set_title("Loss Curves — CLIP-L/14@336"); ax.legend(); ax.grid(True)

            ax = axes[1]
            ax.plot(vis_ratio_log, color="darkorange", linewidth=1.5)
            anneal_end = int(total_steps * anneal_fraction)
            ax.axvline(anneal_end, color="black", linestyle=":",
                       label=f"Anneal end (step {anneal_end})")
            ax.axhline(0.0, color="red", linestyle="--", label="Inference (0.0)")
            ax.set_xlabel("Step"); ax.set_ylabel("Visible GT Patch Ratio")
            ax.set_title("Patch Annealing Schedule"); ax.legend(); ax.grid(True)

            ax = axes[2]
            window  = 100
            rolling = [
                sum(mask_ratio_log[max(0, i-window): i+1]) /
                len(mask_ratio_log[max(0, i-window): i+1])
                for i in range(len(mask_ratio_log))
            ]
            ax.plot(mask_ratio_log, alpha=0.12, color="purple", label="Sampled")
            ax.plot(rolling, color="purple", linewidth=2,
                    label=f"Rolling avg ({window})")
            ax.axhline(1.00, color="red", linestyle="--", label="Inference (1.0)")
            ax.set_xlabel("Step"); ax.set_ylabel("Mask Ratio")
            ax.set_title("Dynamic Masking Distribution")
            ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True)

            plt.tight_layout()
            plt.savefig("jepa_training_curve_v2.png", dpi=150)
            plt.close()

    # ── SAVE CHECKPOINT ───────────────────────────────────────────────────
    # Saved to scratch (large quota) not home (small quota).
    # Keys: text_encoder, patch_encoder, predictor, config.
    # NO loss_head key — it no longer exists in this architecture.
    save_path = "/scratch/knishant_iitp/jepa_medical_v2.pth"
    torch.save(
        {
            "text_encoder":  text_encoder.state_dict(),
            "patch_encoder": patch_encoder.state_dict(),
            "predictor":     predictor.state_dict(),
            "config": {
                "text_dim":          TEXT_DIM,          # 4096
                "hidden_dim":        HIDDEN_DIM,        # 1024
                "clip_dim":          CLIP_DIM,          # 1024 (= hidden_dim now)
                "num_queries":       NUM_VIS_TOKENS,    # 576
                "mask_ratio_min":    mask_ratio_min,
                "mask_ratio_max":    mask_ratio_max,
                "initial_vis_ratio": initial_vis_ratio,
                "anneal_fraction":   anneal_fraction,
                "lambda_refine":     lambda_refine,
                "total_steps":       total_steps,
                "clip_id":           CLIP_ID,
                "pmc_llama_id":      PMC_LLAMA_ID,
            },
        },
        save_path,
    )
    print(f"\nTraining complete. Checkpoint saved → {save_path}")


if __name__ == "__main__":
    execute_training_pipeline()
