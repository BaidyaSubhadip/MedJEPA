"""
jepa_train_medical_v3.py  —  Cross-Modal JEPA  (Medical, PMC-VQA)
==================================================================

CHANGES FROM v2
───────────────
Added three complementary loss components on top of existing Smooth-L1:

LOSS 1 — SigLIP (global semantic alignment with negative pairs)
  Replaces softmax CLIP loss with sigmoid binary loss.
  Applied between mean-pooled TEXT context and mean-pooled GT CLIP tokens.
  Uses a momentum memory bank of past embeddings as negatives so it
  works with batch_size=1 (no need to change batch size).

  L_siglip = sigmoid binary CE over (text_global, gt_visual_global) pairs
  Positives: current sample's text↔visual pair
  Negatives: text↔visual pairs from memory bank (last K samples)

  Why SigLIP over InfoNCE:
    InfoNCE normalises via softmax across the batch → needs large batches.
    SigLIP uses sigmoid → each pair scored independently → works with
    small batches and memory banks. Each pair gets its own gradient.

LOSS 2 — Alignment Loss (text↔visual space pull)
  Directly trains TextContextEncoder to produce text representations
  that are close to the CLIP visual space before the predictor runs.
  Applied between mean-pooled text_ctx and mean-pooled GT CLIP tokens.
  Gradient flows into TextContextEncoder only — gives it a direct
  training signal to cross the text↔visual space gap.

  L_align = 1 - cosine_similarity(mean(text_ctx), mean(gt_clip))

LOSS 3 — VICReg Variance Term (anti-collapse regulariser)
  Prevents representation collapse — the most common failure mode
  in self-supervised learning where the predictor outputs the same
  average vector regardless of input.

  L_var = mean( max(0, γ - std(pred_tokens, dim=0)) )
  Forces per-dimension standard deviation ≥ γ (default 1.0).
  Applied to the predictor output tokens, not to loss_head.

UNCHANGED FROM v2
──────────────────
  L_smooth  : Smooth-L1 on L2-normalised predicted vs GT tokens (primary)
  L_refine  : Self-refinement pass-2 loss (λ=0.5)
  Architecture: TextContextEncoder, PatchContextEncoder, JEPAPredictor
  No LossHead, no out_proj, CLIP-L/14@336 (correct distribution)

TOTAL LOSS
───────────
  L = L_smooth
    + λ_refine × L_refine
    + λ_siglip × L_siglip
    + λ_align  × L_align
    + λ_var    × L_var

DATASET
────────
  TRAINING   : train_2.csv  (rows after first val_size, shuffled seed=42)
  VALIDATION : train_2.csv  (first val_size=300 rows, shuffled seed=42)
  TEST       : test_2.csv   (held-out, never loaded here)
"""

import os
import math
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image

# ── HuggingFace cache — must be set before any HF import ────────────────────
HF_CACHE_DIR = "/scratch/knishant_iitp/"
os.environ["HF_HOME"]               = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]    = HF_CACHE_DIR

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NUM_VIS_TOKENS = 576    # CLIP-ViT-L/14@336  →  24×24 = 576 patches
CLIP_DIM       = 1024   # ViT-L hidden size  =  HIDDEN_DIM  (no mismatch)
TEXT_DIM       = 4096   # PMC-LLaMA 7B embedding dimension
HIDDEN_DIM     = 1024   # JEPA internal dimension

IMAGE_ROOT = "./PMC-VQA/figures/"
TRAIN_CSV  = "./PMC-VQA/train_2.csv"

CLIP_ID      = "openai/clip-vit-large-patch14-336"
PMC_LLAMA_ID = "chaoyi-wu/PMC_LLaMA_7B"


# ══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE  (identical to v2 — no changes)
# ══════════════════════════════════════════════════════════════════════════════

class TextContextEncoder(nn.Module):
    """
    PMC-LLaMA embeddings [B, T, 4096] → context [B, T, 1024].
    input_proj Linear(4096→1024) is the only projection on the text path.
    """
    def __init__(self, text_dim=TEXT_DIM, hidden_dim=HIDDEN_DIM,
                 num_heads=16, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(text_dim, hidden_dim)
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
    Adds learnable positional embeddings to CLIP-L patch tokens.
    No input_proj — CLIP-L dim (1024) = HIDDEN_DIM natively.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, num_patches=NUM_VIS_TOKENS):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patches, patch_indices):
        pos = self.pos_embed[:, patch_indices, :]
        return self.norm(patches + pos)


class JEPAPredictor(nn.Module):
    """
    Cross-attention decoder → predicted visual tokens [B, M, 1024].
    No out_proj — output dim matches CLIP-L targets and LLaVA input.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, num_queries=NUM_VIS_TOKENS,
                 num_heads=8, num_layers=4):
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
#  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def jepa_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    PRIMARY LOSS — unchanged from v2.
    Smooth-L1 between L2-normalised predicted and GT CLIP patch tokens.
    Stop-gradient on target ensures no gradient flows into frozen CLIP.

    predicted: [B, M, 1024]
    target:    [B, M, 1024]
    """
    return F.smooth_l1_loss(
        F.normalize(predicted,       dim=-1),
        F.normalize(target.detach(), dim=-1),
    )


def alignment_loss(
    text_ctx:    torch.Tensor,    # [B, T, 1024]  text context
    gt_clip:     torch.Tensor,    # [B, N, 1024]  GT CLIP patch tokens
) -> torch.Tensor:
    """
    ALIGNMENT LOSS — directly trains TextContextEncoder to produce text
    representations that are close to the CLIP visual space.

    Mean-pools both sides → computes 1 - cosine_similarity.

    This gives TextContextEncoder a direct gradient signal to cross
    the PMC-LLaMA↔CLIP gap, which the JEPA loss alone cannot do
    efficiently (it only signals through the predictor).

    text_ctx : [B, T, 1024]  — output of TextContextEncoder
    gt_clip  : [B, N, 1024]  — GT CLIP patch tokens (stop-grad applied here)

    Returns scalar loss in [0, 2].
    A perfect text→visual bridge gives 0. Orthogonal spaces give 1.
    """
    # Mean-pool: [B, 1024]
    text_global = text_ctx.mean(dim=1)             # [B, 1024]
    clip_global = gt_clip.detach().mean(dim=1)     # [B, 1024]  stop-grad

    # L2 normalise
    text_norm = F.normalize(text_global, dim=-1)   # [B, 1024]
    clip_norm = F.normalize(clip_global, dim=-1)   # [B, 1024]

    # Cosine similarity → loss = 1 - sim  (0 = perfect, 2 = opposite)
    cos_sim = (text_norm * clip_norm).sum(dim=-1)  # [B]
    return (1.0 - cos_sim).mean()


class SigLIPLoss(nn.Module):
    """
    SIGLIP LOSS — global semantic alignment via sigmoid binary cross-entropy.

    Unlike InfoNCE (softmax), SigLIP scores each (text, visual) pair
    independently via sigmoid — no batch-level normalisation.
    This means it works with small effective batch sizes.

    To provide negative pairs with batch_size=1, we maintain a
    MOMENTUM MEMORY BANK — a rolling deque of past GT visual embeddings.
    Each step we have 1 positive pair + K negative pairs from the bank.

    Formulation (per Zhai et al. 2023):
      logit_ij = (text_i · visual_j) / τ + b
      L = -1/N × Σ_ij [ y_ij × log(σ(logit_ij))
                       + (1-y_ij) × log(1 - σ(logit_ij)) ]
      y_ij = 1 if i==j (positive), 0 otherwise (negative)

    Parameters
    ──────────
    memory_size : int
        How many past visual embeddings to keep as negatives.
        Minimum meaningful: 8. Recommended: 64-128.
        More = more stable contrastive signal, more memory.

    temperature : float
        Scaling factor for logits. Lower = harder negatives.
        SigLIP paper found 1/τ ~ 10 (i.e. τ ~ 0.1) works well.

    init_bias : float
        Initial value of the learned bias term.
        SigLIP paper uses -10 which initialises probabilities near zero,
        preventing false positive signals early in training.
    """

    def __init__(
        self,
        memory_size: int   = 64,
        temperature: float = 0.07,
        init_bias:   float = -10.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.bias        = nn.Parameter(torch.tensor(init_bias))

        # Memory bank: rolling deque of normalised GT visual embeddings [1024]
        # deque with maxlen auto-drops oldest when full
        self.memory_bank: deque = deque(maxlen=memory_size)

    def forward(
        self,
        text_ctx: torch.Tensor,     # [B, T, 1024]  text context (trained)
        gt_clip:  torch.Tensor,     # [B, N, 1024]  GT CLIP tokens (stop-grad)
    ) -> torch.Tensor:
        """
        Compute SigLIP loss between text global rep and GT visual global rep.

        Gradient flows into:
          - text_ctx (through TextContextEncoder)
          - temperature and bias parameters
        Stop-gradient applied to gt_clip so CLIP encoder stays frozen.
        """
        device = text_ctx.device

        # ── Global representations (mean-pooled, L2-normalised) ───────────
        text_global = F.normalize(text_ctx.mean(dim=1), dim=-1)   # [B, 1024]
        vis_global  = F.normalize(
            gt_clip.detach().mean(dim=1), dim=-1                   # [B, 1024]
        )

        B = text_global.shape[0]   # typically 1

        # ── Current positive pairs ─────────────────────────────────────────
        # Positive logit: text_i · vis_i  (same sample)
        pos_logit = (text_global * vis_global).sum(dim=-1)         # [B]
        pos_logit = pos_logit / self.temperature.abs() + self.bias

        # Positive loss: -log(σ(pos_logit))
        loss_pos = -F.logsigmoid(pos_logit).mean()

        # ── Negative pairs from memory bank ───────────────────────────────
        loss_neg = torch.tensor(0.0, device=device)

        if len(self.memory_bank) > 0:
            # Stack memory bank embeddings  [K, 1024]
            bank_embeds = torch.stack(
                list(self.memory_bank), dim=0
            ).to(device)                                            # [K, 1024]

            # Negative logits: text_i · vis_k  for all k in bank
            # vis_k are from DIFFERENT samples → all negatives
            neg_logits = (
                text_global @ bank_embeds.T                         # [B, K]
            ) / self.temperature.abs() + self.bias

            # Negative loss: -log(1 - σ(neg_logit)) = -log(σ(-neg_logit))
            loss_neg = -F.logsigmoid(-neg_logits).mean()

        # ── Update memory bank ────────────────────────────────────────────
        # Add current GT visual embeddings (detached — no gradient into bank)
        for i in range(B):
            self.memory_bank.append(vis_global[i].detach().cpu())

        return loss_pos + loss_neg


class VICRegVarianceLoss(nn.Module):
    """
    VARIANCE TERM from VICReg (Bardes et al. 2022).
    Prevents representation collapse by penalising low variance.

    Collapse happens when the predictor learns to output the same
    average token regardless of input text — this is undetectable
    from Smooth-L1 alone if the dataset mean happens to be close
    to many GT tokens.

    L_var = mean_d( max(0, γ - std_n(z_d)) )

    For each feature dimension d, the standard deviation across
    the batch of N predictions must be ≥ γ. If it falls below,
    we penalise proportionally.

    Here we apply it across the 576 token predictions within
    a single sample (spatial variance), which works with batch=1.

    gamma : float
        Target minimum standard deviation per dimension.
        VICReg paper uses 1.0. Start conservative at 0.5.
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred : [B, N, D]  — predictor output tokens
               N = 576 patch tokens (used as the "batch" for variance)
               D = 1024 feature dimensions

        Computes std across the N=576 token dimension for each
        feature dimension, then penalises any dim below gamma.
        """
        # pred: [B, N, D] → compute std across N tokens → [B, D]
        # Use unbiased std (ddof=1) — N=576 is large enough
        std = pred.std(dim=1)                          # [B, D]

        # Penalise dimensions with std < gamma
        variance_loss = F.relu(self.gamma - std).mean()
        return variance_loss


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING MECHANISM HELPERS  (unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def get_visible_ratio(step, total_steps, initial_ratio=0.50, anneal_fraction=0.80):
    anneal_steps = int(total_steps * anneal_fraction)
    if step >= anneal_steps:
        return 0.0
    return initial_ratio * 0.5 * (1.0 + math.cos(math.pi * step / anneal_steps))


def build_full_context(text_ctx, patches, patch_indices, patch_encoder):
    if patches is None or patch_indices is None or patches.shape[1] == 0:
        return text_ctx
    patch_ctx = patch_encoder(patches, patch_indices)
    return torch.cat([text_ctx, patch_ctx], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_caption(row):
    caption = str(row.get("Caption", "")).strip()
    return caption if caption else str(row.get("Question", "")).strip()


def load_image_safe(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def prepare_dataframe(csv_path, val_size, seed=42):
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
    print(f"  train: {len(train_df):,}  |  val: {len(val_df):,}")
    print(f"  test_2.csv is held-out — not loaded here")
    return train_df, val_df


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def execute_training_pipeline(
    total_steps:       int   = 40_000,   # increased from 20k — more signal needed
    # ── Patch annealing ──────────────────────────────────────────────────
    initial_vis_ratio: float = 0.50,
    anneal_fraction:   float = 0.80,
    # ── Dynamic masking ──────────────────────────────────────────────────
    mask_ratio_min:    float = 0.50,
    mask_ratio_max:    float = 1.00,
    # ── Loss weights ─────────────────────────────────────────────────────
    lambda_refine:     float = 0.50,   # refinement pass-2 weight
    lambda_siglip:     float = 0.20,   # SigLIP global alignment weight
    lambda_align:      float = 0.10,   # direct text↔visual alignment weight
    lambda_var:        float = 0.15,   # VICReg variance anti-collapse weight
    # ── SigLIP memory bank ───────────────────────────────────────────────
    siglip_memory:     int   = 128,     # past samples kept as negatives
    siglip_temp:       float = 0.07,   # initial temperature
    siglip_bias:       float = -10.0,  # initial bias (SigLIP paper default)
    vicreg_gamma:      float = 0.5,    # min std per feature dim
    # ── Gradient accumulation ────────────────────────────────────────────
    accum_steps:       int   = 8,      # effective batch = accum_steps × 1
    # ── Optimisation ─────────────────────────────────────────────────────
    val_size:          int   = 300,
    val_every:         int   = 200,
    lr:                float = 5e-5,
    weight_decay:      float = 0.05,
    max_text_len:      int   = 128,
):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  JEPA Medical Training  v3  (SigLIP + Align + VICReg)")
    print("=" * 65)
    print(f"  Total steps        : {total_steps}")
    print(f"  Effective batch    : {accum_steps}  (gradient accumulation)")
    print(f"  Loss weights       : smooth_l1=1.0  refine={lambda_refine}")
    print(f"                       siglip={lambda_siglip}  align={lambda_align}  var={lambda_var}")
    print(f"  SigLIP memory bank : {siglip_memory} samples")
    print(f"  VICReg γ           : {vicreg_gamma}")
    print("=" * 65)

    # ── 1. FROZEN CLIP (target encoder) ───────────────────────────────────
    print(f"\nLoading CLIP: {CLIP_ID} ...")
    clip_processor = CLIPImageProcessor.from_pretrained(
        CLIP_ID, cache_dir=HF_CACHE_DIR
    )
    clip_model = CLIPVisionModel.from_pretrained(
        CLIP_ID, cache_dir=HF_CACHE_DIR
    ).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # ── 2. FROZEN PMC-LLaMA EMBEDDING LAYER ───────────────────────────────
    print(f"Loading PMC-LLaMA: {PMC_LLAMA_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        PMC_LLAMA_ID, use_fast=False, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        PMC_LLAMA_ID, torch_dtype=torch.float16,
        device_map="cpu", cache_dir=HF_CACHE_DIR,
    )
    text_embed_layer = llm.get_input_embeddings().to(device)
    text_embed_layer.eval()
    for p in text_embed_layer.parameters():
        p.requires_grad_(False)
    del llm
    torch.cuda.empty_cache()

    # ── 3. LEARNABLE JEPA COMPONENTS ──────────────────────────────────────
    text_encoder  = TextContextEncoder().to(device)
    patch_encoder = PatchContextEncoder().to(device)
    predictor     = JEPAPredictor().to(device)

    # ── 4. LEARNABLE LOSS COMPONENTS ──────────────────────────────────────
    siglip_loss_fn = SigLIPLoss(
        memory_size = siglip_memory,
        temperature = siglip_temp,
        init_bias   = siglip_bias,
    ).to(device)

    vicreg_loss_fn = VICRegVarianceLoss(gamma=vicreg_gamma)

    # All trainable parameters including SigLIP temperature + bias
    all_params = (
        list(text_encoder.parameters())     +
        list(patch_encoder.parameters())    +
        list(predictor.parameters())        +
        list(siglip_loss_fn.parameters())   # temperature + bias are learned
    )
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in all_params) / 1e6:.1f}M")

    # ── 5. OPTIMISER + SCHEDULE ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    def lr_lambda(step):
        warmup = 1500
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── 6. DATASET ────────────────────────────────────────────────────────
    print(f"\nLoading {TRAIN_CSV} ...")
    train_df, val_df = prepare_dataframe(TRAIN_CSV, val_size)
    train_cursor = 0

    # ── 7. PRE-COMPUTE VALIDATION SET ────────────────────────────────────
    print(f"Pre-computing {len(val_df)} validation samples ...")
    val_data = []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            img_path = os.path.join(IMAGE_ROOT, row["Figure_path"])
            image    = load_image_safe(img_path)
            if image is None:
                continue
            pv = clip_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device)
            clip_out    = clip_model(pv, output_hidden_states=True)
            clip_tokens = clip_out.hidden_states[-2][:, 1:, :]  # [1, 576, 1024]
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
    print(f"Validation set: {len(val_data)} samples")

    # ── 8. TRACKING VARIABLES ─────────────────────────────────────────────
    train_losses   = []
    # Track each loss component separately for diagnosis
    smooth_log, refine_log, siglip_log, align_log, var_log = [], [], [], [], []
    val_losses_p1  = []
    val_losses_p2  = []
    val_steps      = []
    mask_ratio_log = []
    vis_ratio_log  = []

    # ── 9. TRAINING LOOP ──────────────────────────────────────────────────
    print(f"\nStarting training for {total_steps} steps "
          f"(effective batch={accum_steps}) ...\n")

    optimizer.zero_grad()   # reset before accumulation loop
    accum_count    = 0
    accum_loss_buf = {
        "smooth": 0.0, "refine": 0.0,
        "siglip": 0.0, "align":  0.0, "var": 0.0,
    }

    for step in range(total_steps):
        text_encoder.train()
        patch_encoder.train()
        predictor.train()

        row = train_df.iloc[train_cursor % len(train_df)]
        train_cursor += 1

        img_path = os.path.join(IMAGE_ROOT, row["Figure_path"])
        image    = load_image_safe(img_path)
        if image is None:
            mask_ratio_log.append(0.75)
            vis_ratio_log.append(get_visible_ratio(step, total_steps))
            continue

        # ── Extract frozen features ────────────────────────────────────────
        with torch.no_grad():
            pv = clip_processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device)
            clip_out    = clip_model(pv, output_hidden_states=True)
            clip_tokens = clip_out.hidden_states[-2][:, 1:, :]  # [1, 576, 1024]

            caption      = build_caption(row)
            tok = tokenizer(
                caption, return_tensors="pt",
                truncation=True, max_length=max_text_len,
                padding="max_length",
            ).to(device)
            raw_text_emb = text_embed_layer(tok.input_ids).float()

        # ── Step A: Encode text ────────────────────────────────────────────
        text_ctx = text_encoder(raw_text_emb, tok.attention_mask)  # [1, T, 1024]

        # ── Step B: Patch annealing ────────────────────────────────────────
        vis_ratio   = get_visible_ratio(step, total_steps, initial_vis_ratio, anneal_fraction)
        num_visible = int(NUM_VIS_TOKENS * vis_ratio)
        vis_ratio_log.append(vis_ratio)

        if num_visible > 0:
            perm        = torch.randperm(NUM_VIS_TOKENS, device=device)
            visible_idx = perm[:num_visible].sort().values
            mask_pool   = perm[num_visible:].sort().values
        else:
            visible_idx = None
            mask_pool   = torch.arange(NUM_VIS_TOKENS, device=device)

        # ── Step C: Dynamic masking ────────────────────────────────────────
        mask_ratio  = random.uniform(mask_ratio_min, mask_ratio_max)
        num_masked  = min(len(mask_pool), max(1, int(len(mask_pool) * mask_ratio)))
        sub_perm    = torch.randperm(len(mask_pool), device=device)
        masked_idx  = mask_pool[sub_perm[:num_masked]].sort().values
        mask_ratio_log.append(mask_ratio)

        # ── Step D: Build context ──────────────────────────────────────────
        vis_patches = (
            clip_tokens[:, visible_idx, :] if visible_idx is not None else None
        )
        full_ctx = build_full_context(
            text_ctx, vis_patches, visible_idx, patch_encoder
        )

        # ── Step E: Primary JEPA loss (Smooth-L1) ─────────────────────────
        predicted  = predictor(full_ctx, mask_indices=masked_idx)  # [1, M, 1024]
        target     = clip_tokens[:, masked_idx, :]                 # [1, M, 1024]
        loss_smooth = jepa_loss(predicted, target)

        # ── Step F: Self-refinement loss ──────────────────────────────────
        with torch.no_grad():
            pred_pass1 = predictor(text_ctx)   # [1, 576, 1024]
            all_idx    = torch.arange(NUM_VIS_TOKENS, device=device)
        refine_ctx  = build_full_context(
            text_ctx, pred_pass1.detach(), all_idx, patch_encoder
        )
        pred_pass2  = predictor(refine_ctx, mask_indices=masked_idx)
        loss_refine = jepa_loss(pred_pass2, target)

        # ── Step G: Alignment loss ─────────────────────────────────────────
        # Directly trains TextContextEncoder to produce representations
        # close to CLIP visual space. Gradient goes into text_encoder only.
        # clip_tokens stop-gradient applied inside alignment_loss.
        loss_align = alignment_loss(text_ctx, clip_tokens)

        # ── Step H: SigLIP loss ────────────────────────────────────────────
        # Global semantic alignment between text and GT visual reps.
        # Uses memory bank of past GT visual embeddings as negatives.
        # Gradient flows into text_encoder + temperature/bias params.
        loss_siglip = siglip_loss_fn(text_ctx, clip_tokens)

        # ── Step I: VICReg variance loss ───────────────────────────────────
        # Applied to full 576-token predictor output (not just masked subset).
        # Prevents predictor from collapsing to outputting the same token.
        # Use pred_pass1 which covers all 576 positions.
        loss_var = vicreg_loss_fn(pred_pass1)   # pred_pass1 is [1, 576, 1024]

        # ── Step J: Total loss ─────────────────────────────────────────────
        loss = (
            loss_smooth
            + lambda_refine  * loss_refine
            + lambda_siglip  * loss_siglip
            + lambda_align   * loss_align
            + lambda_var     * loss_var
        )

        # ── Step K: Gradient accumulation ─────────────────────────────────
        # Accumulate over accum_steps before updating weights.
        # Simulates effective batch size = accum_steps.
        # Divide loss by accum_steps so total magnitude is normalised.
        (loss / accum_steps).backward()
        accum_count += 1

        accum_loss_buf["smooth"] += loss_smooth.item() / accum_steps
        accum_loss_buf["refine"] += loss_refine.item() / accum_steps
        accum_loss_buf["siglip"] += loss_siglip.item() / accum_steps
        accum_loss_buf["align"]  += loss_align.item()  / accum_steps
        accum_loss_buf["var"]    += loss_var.item()    / accum_steps

        if accum_count == accum_steps:
            # Actual weight update (every accum_steps raw steps)
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log accumulated loss
            total_accum = (
                accum_loss_buf["smooth"]
                + lambda_refine * accum_loss_buf["refine"]
                + lambda_siglip * accum_loss_buf["siglip"]
                + lambda_align  * accum_loss_buf["align"]
                + lambda_var    * accum_loss_buf["var"]
            )
            train_losses.append(total_accum)
            smooth_log.append(accum_loss_buf["smooth"])
            refine_log.append(accum_loss_buf["refine"])
            siglip_log.append(accum_loss_buf["siglip"])
            align_log.append(accum_loss_buf["align"])
            var_log.append(accum_loss_buf["var"])

            # Reset accumulation buffer
            accum_count = 0
            accum_loss_buf = {k: 0.0 for k in accum_loss_buf}

        else:
            # Mid-accumulation — just record raw loss for logging
            train_losses.append(loss.item())

        # ── VALIDATION ────────────────────────────────────────────────────
        if (step + 1) % val_every == 0:
            text_encoder.eval()
            patch_encoder.eval()
            predictor.eval()

            total_vl_p1  = 0.0
            total_vl_p2  = 0.0
            total_vl_cos = 0.0  # track cosine sim directly during val
            all_idx_val  = torch.arange(NUM_VIS_TOKENS, device=device)

            with torch.no_grad():
                for vd in val_data:
                    v_clip = vd["clip_tokens"].to(device)
                    v_text = vd["raw_text_emb"].to(device)
                    v_mask = vd["attention_mask"].to(device)

                    v_text_ctx = text_encoder(v_text, v_mask)

                    v_pred_p1 = predictor(v_text_ctx)
                    total_vl_p1 += jepa_loss(v_pred_p1, v_clip).item()

                    # Cosine similarity (direct representation quality)
                    cos = (
                        F.normalize(v_pred_p1, dim=-1) *
                        F.normalize(v_clip, dim=-1)
                    ).sum(dim=-1).mean().item()
                    total_vl_cos += cos

                    v_refine_ctx = build_full_context(
                        v_text_ctx, v_pred_p1, all_idx_val, patch_encoder
                    )
                    v_pred_p2 = predictor(v_refine_ctx)
                    total_vl_p2 += jepa_loss(v_pred_p2, v_clip).item()

            n_val       = len(val_data)
            avg_vl_p1   = total_vl_p1  / n_val
            avg_vl_p2   = total_vl_p2  / n_val
            avg_cos_sim = total_vl_cos  / n_val
            curr_vis    = get_visible_ratio(step, total_steps)
            curr_lr     = scheduler.get_last_lr()[0]

            val_losses_p1.append(avg_vl_p1)
            val_losses_p2.append(avg_vl_p2)
            val_steps.append(step + 1)

            # Current loss breakdown (last logged values)
            last_s = smooth_log[-1]  if smooth_log  else 0.0
            last_r = refine_log[-1]  if refine_log  else 0.0
            last_g = siglip_log[-1]  if siglip_log  else 0.0
            last_a = align_log[-1]   if align_log    else 0.0
            last_v = var_log[-1]     if var_log      else 0.0

            print(
                f"Step {step+1:>6}/{total_steps} | "
                f"smooth={last_s:.5f} ref={last_r:.5f} "
                f"sig={last_g:.4f} aln={last_a:.4f} var={last_v:.4f} | "
                f"Val P1={avg_vl_p1:.5f} P2={avg_vl_p2:.5f} | "
                f"cos_sim={avg_cos_sim:.4f} | "
                f"vis={curr_vis:.3f} LR={curr_lr:.2e}"
            )

            # ── 4-panel training plot ──────────────────────────────────────
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))

            ax = axes[0]
            ax.plot(train_losses, alpha=0.3, color="steelblue", label="Train (total)")
            ax.plot(val_steps, val_losses_p1, color="red", linewidth=2, label="Val P1")
            ax.plot(val_steps, val_losses_p2, color="green", linewidth=2,
                    linestyle="--", label="Val P2")
            ax.set_title("Loss Curves"); ax.legend(); ax.grid(True)

            ax = axes[1]
            if smooth_log:
                ax.plot(smooth_log,  label="smooth_l1", color="steelblue")
                ax.plot(refine_log,  label="refine",    color="orange")
                ax.plot(siglip_log,  label="siglip",    color="purple")
                ax.plot(align_log,   label="align",     color="green")
                ax.plot(var_log,     label="var",       color="red")
            ax.set_title("Loss Components (per component)"); ax.legend(); ax.grid(True)

            ax = axes[2]
            ax.plot(val_steps, [v/len(val_data)*len(val_data)
                                for v in val_losses_p1],
                    color="red", label="Val cos sim (approx)")
            # Approximate cosine sim from loss
            ax.set_title("Val Loss P1 vs P2"); ax.legend(); ax.grid(True)

            ax = axes[3]
            ax.plot(vis_ratio_log, color="darkorange", label="vis_ratio")
            ax.axhline(0.0, color="red", linestyle="--", label="Inference (0.0)")
            ax.set_title("Patch Annealing"); ax.legend(); ax.grid(True)

            plt.tight_layout()
            plt.savefig("jepa_training_curve_v3.png", dpi=150)
            plt.close()

    # ── SAVE CHECKPOINT ───────────────────────────────────────────────────
    save_path = "/scratch/knishant_iitp/jepa_medical_v3.pth"
    torch.save(
        {
            "text_encoder":  text_encoder.state_dict(),
            "patch_encoder": patch_encoder.state_dict(),
            "predictor":     predictor.state_dict(),
            # Save SigLIP params so temperature + bias are restored
            "siglip_temperature": siglip_loss_fn.temperature.item(),
            "siglip_bias":        siglip_loss_fn.bias.item(),
            "config": {
                "text_dim":          TEXT_DIM,
                "hidden_dim":        HIDDEN_DIM,
                "clip_dim":          CLIP_DIM,
                "num_queries":       NUM_VIS_TOKENS,
                "mask_ratio_min":    mask_ratio_min,
                "mask_ratio_max":    mask_ratio_max,
                "initial_vis_ratio": initial_vis_ratio,
                "anneal_fraction":   anneal_fraction,
                "lambda_refine":     lambda_refine,
                "lambda_siglip":     lambda_siglip,
                "lambda_align":      lambda_align,
                "lambda_var":        lambda_var,
                "total_steps":       total_steps,
                "accum_steps":       accum_steps,
                "clip_id":           CLIP_ID,
                "pmc_llama_id":      PMC_LLAMA_ID,
            },
        },
        save_path,
    )
    print(f"\nTraining complete. Saved → {save_path}")


if __name__ == "__main__":
    execute_training_pipeline()
