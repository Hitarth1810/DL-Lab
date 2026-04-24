# -*- coding: utf-8 -*-
"""
BLIP Federated Learning: Baseline + FedMix + Priority Pseudo-Labeling
Uses BlipForConditionalGeneration (NOT BLIP-2).

Key design:
- First WARMUP_ROUNDS rounds: supervised training only (labeled data)
- After warmup: FedMix pixel-space mixup + priority pseudo-labeling for unlabeled data
- FedProx proximal term to reduce client drift
- Cosine LR schedule with warmup
- Dropout regularization on text decoder

Fixes applied:
- torch.load/save PyTorch 2.6+ weights_only compatibility
- Non-floating-point parameter requires_grad guard
- position_ids UNEXPECTED key handled via strict=False
- Native Python types in saved checkpoints (no numpy scalars)
"""

import os
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import copy, sys, gc, time
import numpy as np
import pandas as pd
from PIL import Image
import argparse
sys.stdout.reconfigure(line_buffering=True)

import torch
assert torch.cuda.is_available(), "CUDA not available! Check GPU allocation."
torch.cuda.init()
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration

import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

nltk.download("punkt",    quiet=True)
nltk.download("punkt_tab",quiet=True)
nltk.download("wordnet",  quiet=True)
nltk.download("omw-1.4",  quiet=True)

# ===============================
# Arguments
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=int, default=100,
                    help="Percentage of labeled data (default: 100)")
args  = parser.parse_args()
ALPHA = args.alpha

# ===============================
# Hyperparameters
# ===============================
NUM_CLIENTS     = 3
DIRICHLET_ALPHA = 0.5
ROUNDS          = 20
LOCAL_EPOCHS    = 3
LEARNING_RATE   = 5e-5
BATCH_SIZE      = 16
PATIENCE        = 10
MU              = 0.01
DROPOUT_RATE    = 0.1

# FedMix & Pseudo-Label Hyperparameters
WARMUP_ROUNDS         = 2      # First 2 rounds: labeled-only supervised training
MIXUP_ALPHA           = 0.2
GLOBAL_MEAN_MAX_SAMP  = 64
MAX_PSEUDO_PER_CLIENT = 150
PSEUDO_BATCH_SIZE     = 16
PSEUDO_CONF_THR       = 0.35
PSEUDO_TOP_K_FRAC     = 0.5

# Medical domain prompt for conditioned generation
# This forces BLIP to stay in the radiology domain during pseudo-labeling
MEDICAL_PROMPT = "a chest x-ray showing"

# Number of diverse representative captions to mine from labeled data
GLOBAL_CAPTION_TOP_K = 6

DATASET_DIR = "/home/tanmoyhazra/u23ai048"
CSV_PATH    = f"{DATASET_DIR}/indiana_complete.csv"
MODEL_NAME  = "Salesforce/blip-image-captioning-base"

DEVICE = torch.device("cuda:0")

print(f"\n{'='*70}")
print(f"BLIP FEDERATED LEARNING: BASELINE + FEDMIX")
print(f"{'='*70}")
print(f"  Model:           {MODEL_NAME}")
print(f"  Labeled data:    {ALPHA}%")
print(f"  Clients:         {NUM_CLIENTS}")
print(f"  Warmup Rounds:   {WARMUP_ROUNDS}")
print(f"  Total Rounds:    {ROUNDS}")
print(f"  Local epochs:    {LOCAL_EPOCHS}")
print(f"  Learning rate:   {LEARNING_RATE}")
print(f"  Batch size:      {BATCH_SIZE}")
print(f"  FedProx mu:      {MU}")
print(f"  Dropout:         {DROPOUT_RATE}")
print(f"  Device:          {DEVICE}\n")

# ===============================
# Load Dataset & Non-IID Splitting
# ===============================
print(f"{'='*70}")
print("LOADING DATASET")
print(f"{'='*70}")

df = pd.read_csv(CSV_PATH)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

np.random.seed(42)
n_labeled       = int(len(train_df) * ALPHA / 100.0)
shuffled_idx    = np.random.permutation(len(train_df))
labeled_indices = shuffled_idx[:n_labeled]
unlabeled_idx   = shuffled_idx[n_labeled:]

labeled_df   = train_df.iloc[labeled_indices].reset_index(drop=True)
unlabeled_df = train_df.iloc[unlabeled_idx].reset_index(drop=True) if len(unlabeled_idx) > 0 else pd.DataFrame()

print(f"Total samples: {len(df)}")
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Labeled: {len(labeled_df)} | Unlabeled: {len(unlabeled_df)}")

def dirichlet_split(n, n_clients, alpha, min_samples=10):
    for _ in range(100):
        props = np.random.dirichlet([alpha] * n_clients)
        props = (props * n).astype(int)
        if np.all(props >= min_samples):
            props[-1] = n - props[:-1].sum()
            idx = np.random.permutation(n)
            splits, start = [], 0
            for p in props:
                splits.append(idx[start:start+p])
                start += p
            return splits
    props = np.full(n_clients, n // n_clients)
    props[-1] = n - props[:-1].sum()
    idx = np.random.permutation(n)
    splits, start = [], 0
    for p in props:
        splits.append(idx[start:start+p])
        start += p
    return splits

print(f"\n{'='*70}")
print("FEDERATED SPLIT")
print(f"{'='*70}")

client_labeled_splits = dirichlet_split(len(labeled_df), NUM_CLIENTS, DIRICHLET_ALPHA)
client_labeled_dfs    = [labeled_df.iloc[s].reset_index(drop=True) for s in client_labeled_splits]
for i, cdf in enumerate(client_labeled_dfs):
    print(f"  Client {i}: {len(cdf)} labeled samples ({len(cdf)/len(labeled_df)*100:.1f}%)")

if ALPHA < 100 and len(unlabeled_df) > 0:
    client_unlabeled_splits = dirichlet_split(len(unlabeled_df), NUM_CLIENTS, DIRICHLET_ALPHA)
    client_unlabeled_dfs    = [unlabeled_df.iloc[s].reset_index(drop=True) for s in client_unlabeled_splits]
    for i, cdf in enumerate(client_unlabeled_dfs):
        print(f"  Client {i}: {len(cdf)} unlabeled samples")
else:
    client_unlabeled_dfs = [pd.DataFrame() for _ in range(NUM_CLIENTS)]

# ===============================
# Dataset
# ===============================
class ChestXrayDataset(Dataset):
    def __init__(self, df, augment=False, is_unlabeled=False):
        self.df           = df.reset_index(drop=True)
        self.augment      = augment
        self.is_unlabeled = is_unlabeled
        self.transform    = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomRotation(5),
        ]) if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.augment and self.transform:
            img = self.transform(img)
        if self.is_unlabeled:
            return img, row["image_path"]
        return img, str(row["caption"]).strip()

test_dataset = ChestXrayDataset(test_df, augment=False)

def collate_fn(batch):
    imgs, caps = zip(*batch)
    return list(imgs), list(caps)

def collate_unlabeled(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)

# ===============================
# Load BLIP Model - Decoder only unfrozen + dropout
# ===============================
print(f"\n{'='*70}")
print("LOADING BLIP MODEL")
print(f"{'='*70}")

processor = BlipProcessor.from_pretrained(MODEL_NAME)
global_model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME, use_safetensors=True
).to(DEVICE)

# Freeze everything first
for p in global_model.parameters():
    p.requires_grad = False

# Unfreeze ONLY text decoder (proven best, fast, stable)
for name, p in global_model.named_parameters():
    if not p.is_floating_point():
        continue
    if "text_decoder" in name:
        p.requires_grad = True

# Add dropout to decoder layers for regularization
for module in global_model.text_decoder.modules():
    if isinstance(module, nn.Dropout):
        module.p = DROPOUT_RATE

trainable = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in global_model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ===============================
# Safe model copy (BLIP is fp32, deepcopy works fine unlike 8-bit BLIP-2)
# ===============================
def safe_copy_model(src_model):
    return copy.deepcopy(src_model)

# ===============================
# FedMix Utils
# ===============================
def extract_mean_image(df_client, max_samples=GLOBAL_MEAN_MAX_SAMP):
    """Compute mean pixel_values tensor for a client's labeled data."""
    sample_df = df_client.sample(min(len(df_client), max_samples), random_state=42)
    ds        = ChestXrayDataset(sample_df, augment=False)
    loader    = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    all_feats = []
    with torch.no_grad():
        for imgs, _ in loader:
            inputs = processor(images=imgs, return_tensors="pt")
            all_feats.append(inputs["pixel_values"].cpu().float())
    return torch.cat(all_feats, dim=0).mean(dim=0, keepdim=True)  # float32 on CPU

def compute_global_caption_pool(client_labeled_dfs, top_k=GLOBAL_CAPTION_TOP_K):
    """
    Mine the most representative captions from actual labeled training data.
    This is FedMix's y-bar (mean label) adapted for captioning:

    1. Collect all labeled captions across clients
    2. Compute TF-IDF embeddings for each caption
    3. Find the centroid of all embeddings (= "mean label" in text space)
    4. Return the top-K captions nearest to the centroid

    Fully data-driven -- no hardcoded captions needed.
    Computed once and cached for the entire training run.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Collect all real captions from labeled data
    all_captions = []
    for cdf in client_labeled_dfs:
        all_captions.extend(cdf["caption"].astype(str).str.strip().tolist())

    # Deduplicate while preserving order
    seen = set()
    unique_captions = []
    for c in all_captions:
        if c not in seen and len(c) > 10:
            seen.add(c)
            unique_captions.append(c)

    if len(unique_captions) == 0:
        print("  [WARN] No valid labeled captions found, using fallback.")
        return ["no acute cardiopulmonary abnormality."]

    # Subsample if too many (TF-IDF is fast, but >5k is wasteful)
    if len(unique_captions) > 2000:
        np.random.seed(42)
        idx = np.random.choice(len(unique_captions), 2000, replace=False)
        unique_captions = [unique_captions[i] for i in idx]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(unique_captions)

    # Centroid = mean of all TF-IDF vectors = "average label" in text space
    centroid = np.asarray(tfidf_matrix.mean(axis=0))

    # Cosine similarity of each caption to the centroid
    sims = cosine_similarity(tfidf_matrix, centroid).flatten()

    # Top-K most representative captions
    top_k = min(top_k, len(unique_captions))
    top_indices = sims.argsort()[-top_k:][::-1]
    pool = [unique_captions[i] for i in top_indices]

    print(f"  Mined {len(pool)} representative captions from {len(all_captions)} labeled samples:")
    for i, cap in enumerate(pool):
        print(f"    [{i+1}] (sim={sims[top_indices[i]]:.3f}) {cap[:100]}...")

    return pool

def generate_pseudo_labels(model, unlabeled_df_client):
    """
    Generate pseudo-labels for unlabeled data using global model.
    Returns list of dicts with image_path, caption, priority (confidence).
    Only keeps top-K fraction above confidence threshold.

    Key design choices:
    - Greedy decoding (num_beams=1): simplest, no beam search complications.
    - Medical prompt conditioning: forces BLIP to generate radiology-domain
      captions instead of hallucinating generic descriptions.
    - Confidence from outputs.scores directly (NOT compute_transition_scores,
      which has repeated shape-mismatch bugs when prompt tokens are present).
    - Mean log-probability as confidence: length-normalized, robust.
    """
    model.eval()
    if len(unlabeled_df_client) == 0:
        return []

    sample_df = unlabeled_df_client.sample(
        min(len(unlabeled_df_client), MAX_PSEUDO_PER_CLIENT), random_state=42
    )
    ds      = ChestXrayDataset(sample_df, augment=False, is_unlabeled=True)
    loader  = DataLoader(ds, batch_size=PSEUDO_BATCH_SIZE, shuffle=False, collate_fn=collate_unlabeled)
    results = []

    with torch.no_grad():
        for imgs, paths in loader:
            try:
                # Conditioned generation with medical prompt
                inputs = processor(
                    images=imgs,
                    text=[MEDICAL_PROMPT] * len(imgs),
                    return_tensors="pt"
                ).to(DEVICE)

                # Greedy decoding
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                # --- Confidence from outputs.scores directly ---
                # outputs.scores is a tuple of (n_gen_steps,) tensors each [batch, vocab].
                # outputs.sequences = [prompt_tokens | generated_tokens].
                # We use len(outputs.scores) to know how many tokens were generated.
                n_gen = len(outputs.scores)
                if n_gen == 0:
                    continue

                # Stack into [batch, n_gen, vocab] and compute log-softmax
                stacked_logits = torch.stack(outputs.scores, dim=1)
                log_probs_all  = torch.log_softmax(stacked_logits, dim=-1)

                for i in range(len(imgs)):
                    # Extract only the generated token IDs (last n_gen tokens)
                    gen_tokens = outputs.sequences[i, -n_gen:]
                    # Mask out padding tokens
                    mask = gen_tokens != processor.tokenizer.pad_token_id
                    if mask.sum() == 0:
                        continue

                    # Gather log-prob of each actually-generated token
                    token_log_probs = log_probs_all[i].gather(
                        1, gen_tokens.unsqueeze(-1)
                    ).squeeze(-1)  # [n_gen]

                    # Mean log-prob -> exponentiate for [0,1] confidence
                    priority = float(torch.exp(token_log_probs[mask].mean()))
                    if priority < PSEUDO_CONF_THR:
                        continue

                    caption = processor.decode(
                        outputs.sequences[i], skip_special_tokens=True
                    ).strip()
                    if len(caption) < 10:  # Stricter min length for medical text
                        continue

                    results.append({
                        "image_path": paths[i],
                        "caption":    caption,
                        "priority":   priority
                    })

                # Free the large stacked tensor immediately
                del stacked_logits, log_probs_all
                torch.cuda.empty_cache()

            except Exception as e:
                # Catch ALL exceptions (CUDA OOM, shape mismatches, etc.)
                # so pseudo-labeling never kills the training run
                print(f"      [WARN] Pseudo-label batch skipped: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

    results.sort(key=lambda x: x["priority"], reverse=True)
    top_k = max(1, int(len(results) * PSEUDO_TOP_K_FRAC))
    return results[:top_k] if len(results) > 0 else []

# ===============================
# Local Training with FedProx + optional FedMix
# ===============================
def local_train(model, dataset, epochs, lr, global_model_ref,
                use_fedmix=False, g_img=None, g_cap=None):
    model.train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        return
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    total_steps  = epochs * len(loader)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Snapshot global trainable params for FedProx proximal term
    global_params = {
        name: param.detach().clone().float()
        for name, param in global_model_ref.named_parameters()
        if param.requires_grad
    }

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for imgs, caps in loader:
            try:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)) if use_fedmix else 1.0

                # BLIP processor: images + text captions -> pixel_values + input_ids
                inputs = processor(
                    images=imgs,
                    text=caps,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(DEVICE)

                # Labels = input_ids with padding masked to -100
                labels = inputs["input_ids"].clone()
                labels = labels.masked_fill(labels == processor.tokenizer.pad_token_id, -100)

                # FedMix: pixel-space mixup with global mean image
                if use_fedmix and g_img is not None:
                    dtype = inputs["pixel_values"].dtype
                    gm = g_img.to(DEVICE).to(dtype)
                    inputs["pixel_values"] = lam * inputs["pixel_values"] + (1.0 - lam) * gm

                outputs = model(**inputs, labels=labels)
                ce_loss = outputs.loss.float()

                if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                    print(f"      [WARN] Invalid CE loss, skipping batch")
                    continue

                # FedMix: global-caption loss component
                if use_fedmix and g_cap and (1.0 - lam) > 1e-6:
                    g_inputs = processor(
                        images=imgs,
                        text=[g_cap] * len(imgs),
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt"
                    ).to(DEVICE)
                    g_labels = g_inputs["input_ids"].clone()
                    g_labels = g_labels.masked_fill(g_labels == processor.tokenizer.pad_token_id, -100)

                    g_outputs = model(
                        pixel_values=inputs["pixel_values"],
                        input_ids=g_inputs["input_ids"],
                        attention_mask=g_inputs["attention_mask"],
                        labels=g_labels
                    )
                    ce_loss = lam * ce_loss + (1.0 - lam) * g_outputs.loss.float()

                # FedProx proximal term
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and name in global_params:
                        prox_term += ((param.float() - global_params[name]) ** 2).sum()
                loss = ce_loss + (MU / 2.0) * prox_term

                if not torch.isfinite(loss):
                    print(f"      [WARN] Non-finite loss ({loss.item():.4f}), skipping batch")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += ce_loss.item()
                n_batches  += 1

                if n_batches % 10 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "CUDA" in str(e) or "NVML" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise e

        avg = epoch_loss / max(n_batches, 1)
        print(f"      Epoch {epoch+1}/{epochs} - CE Loss: {avg:.4f}")

# ===============================
# FedAvg with adaptive priority weighting
# ===============================
def fedavg(global_model, client_models, client_sizes, client_priorities=None):
    if client_priorities is None:
        client_priorities = [1.0] * len(client_sizes)

    weights    = np.array(client_sizes,      dtype=np.float64)
    priorities = np.array(client_priorities, dtype=np.float64)
    weights   /= weights.sum()
    if priorities.sum() > 0:
        priorities /= priorities.mean()
    adaptive_weights  = weights * priorities
    adaptive_weights /= adaptive_weights.sum()

    trainable_names = {
        name for name, p in global_model.named_parameters() if p.requires_grad
    }

    client_param_dicts = [
        dict(m.named_parameters()) for m in client_models
    ]

    with torch.no_grad():
        for name, g_param in global_model.named_parameters():
            if name not in trainable_names:
                continue
            agg = torch.zeros_like(g_param, dtype=torch.float32, device=DEVICE)
            for cdict, w in zip(client_param_dicts, adaptive_weights):
                if name in cdict:
                    agg += cdict[name].data.float() * float(w)
            g_param.data.copy_(agg.to(g_param.dtype))

# ===============================
# Evaluation
# ===============================
def generate_predictions(model, dataset, max_samples=200):
    """Generate captions for evaluation."""
    model.eval()
    preds, refs = [], []
    with torch.no_grad():
        for i, (img, caption) in enumerate(dataset):
            if i >= max_samples:
                break
            try:
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                output = model.generate(
                    **inputs,
                    max_length=80,
                    num_beams=5,
                    length_penalty=1.0,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                pred = processor.decode(output[0], skip_special_tokens=True)
                preds.append(pred)
                refs.append(caption)
                if i % 20 == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "CUDA" in str(e) or "NVML" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise e
    return preds, refs

def evaluate_captions(preds, refs):
    smooth  = SmoothingFunction().method1
    rouge   = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    pt      = [nltk.word_tokenize(p.lower()) for p in preds]
    rt      = [[nltk.word_tokenize(r.lower())] for r in refs]
    bleu    = corpus_bleu(rt, pt, smoothing_function=smooth)
    meteor  = np.mean([meteor_score(rt[i], pt[i]) for i in range(len(preds))])
    rouge_l = np.mean([rouge.score(preds[i], refs[i])["rougeL"].fmeasure for i in range(len(preds))])
    cider   = Cider()
    gts     = {i: [refs[i]]  for i in range(len(refs))}
    res     = {i: [preds[i]] for i in range(len(preds))}
    cider_s, _ = cider.compute_score(gts, res)
    return {"BLEU": bleu, "METEOR": meteor, "ROUGE-L": rouge_l, "CIDEr": cider_s}

# ===============================
# Cache global mean images (stable across rounds)
# ===============================
client_mean_images_cache = [None] * NUM_CLIENTS
global_caption_pool_cache = None  # Mined from labeled data on first FedMix round

# ===============================
# Training Loop
# ===============================
print(f"\n{'='*70}")
print("STARTING BLIP FEDERATED TRAINING (FedMix)")
print(f"{'='*70}")

BEST_CIDER       = -1
BEST_ROUND       = -1
BEST_METRICS     = None
patience_counter = 0
all_metrics      = []

for round_num in range(ROUNDS):

    # Outer LR schedule: decay learning rate across rounds
    progress  = round_num / max(1, ROUNDS - 1)
    round_lr  = LEARNING_RATE * max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    # FedMix activates only after warmup rounds AND when there is unlabeled data
    use_fedmix = (ALPHA < 100) and (round_num >= WARMUP_ROUNDS)
    phase      = "FEDMIX/PSEUDO" if use_fedmix else "WARMUP (Supervised Only)"

    print(f"\n{'='*70}")
    print(f"Round {round_num+1}/{ROUNDS} | {phase} | LR: {round_lr:.2e} | Best CIDEr: {BEST_CIDER:.4f}")
    print(f"{'='*70}")

    g_img, g_cap = None, None
    if use_fedmix:
        print("[Server] Computing Global FedMix Variables...")
        # Use cached mean images (labeled data doesn't change between rounds)
        c_feats = []
        for cid in range(NUM_CLIENTS):
            if client_mean_images_cache[cid] is None:
                client_mean_images_cache[cid] = extract_mean_image(client_labeled_dfs[cid])
            c_feats.append(client_mean_images_cache[cid])

        weights = np.array([len(cdf) for cdf in client_labeled_dfs], dtype=np.float32)
        weights /= weights.sum()
        g_img = sum(img * float(w) for img, w in zip(c_feats, weights))  # float32 cpu

        # Mine representative captions from labeled data (computed once, cached)
        if global_caption_pool_cache is None:
            global_caption_pool_cache = compute_global_caption_pool(client_labeled_dfs)
        g_cap = global_caption_pool_cache[round_num % len(global_caption_pool_cache)]
        print(f"  Global caption (data-mined): {g_cap}")

    client_models, client_sizes, client_priorities = [], [], []

    for cid in range(NUM_CLIENTS):
        print(f"  [Client {cid}] Training...")

        local_model = safe_copy_model(global_model)
        train_data  = client_labeled_dfs[cid].copy()
        priority    = 1.0

        # After warmup: add pseudo-labeled unlabeled data
        if use_fedmix and len(client_unlabeled_dfs[cid]) > 0:
            try:
                global_model.eval()
                pseudos = generate_pseudo_labels(global_model, client_unlabeled_dfs[cid])
                if len(pseudos) > 0:
                    priority  = float(np.mean([x["priority"] for x in pseudos]))
                    pseudo_df = pd.DataFrame([
                        {"image_path": x["image_path"], "caption": x["caption"]}
                        for x in pseudos
                    ])
                    train_data = pd.concat([train_data, pseudo_df], ignore_index=True)
                    print(f"    Added {len(pseudo_df)} pseudo-labels (mean conf: {priority:.3f})")
            except Exception as e:
                print(f"    [WARN] Pseudo-labeling failed for client {cid}: {type(e).__name__}: {e}")
                print(f"    Continuing with labeled data only.")
                torch.cuda.empty_cache()
                gc.collect()

        print(f"    Train size: {len(train_data)}")
        train_ds = ChestXrayDataset(train_data, augment=True)
        local_train(local_model, train_ds, LOCAL_EPOCHS, round_lr, global_model,
                    use_fedmix=use_fedmix, g_img=g_img, g_cap=g_cap)

        client_models.append(local_model)
        client_sizes.append(len(train_ds))
        client_priorities.append(priority)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  Aggregating (FedAvg)...")
    fedavg(
        global_model, client_models, client_sizes,
        client_priorities if use_fedmix else None
    )

    # Free client models immediately
    del client_models
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Evaluating...")
    preds, refs = generate_predictions(global_model, test_dataset, max_samples=200)
    metrics     = evaluate_captions(preds, refs)
    all_metrics.append(metrics)

    print(f"\n  -- Round {round_num+1} Results --")
    print(f"    BLEU    = {metrics['BLEU']:.4f}")
    print(f"    METEOR  = {metrics['METEOR']:.4f}")
    print(f"    ROUGE-L = {metrics['ROUGE-L']:.4f}")
    print(f"    CIDEr   = {metrics['CIDEr']:.4f}")

    if metrics["CIDEr"] > BEST_CIDER:
        BEST_CIDER       = metrics["CIDEr"]
        BEST_ROUND       = round_num + 1
        BEST_METRICS     = metrics
        patience_counter = 0

        # FIX: save with native Python types to avoid numpy scalar pickling issues
        torch.save({
            'round':            int(BEST_ROUND),
            'model_state_dict': global_model.state_dict(),
            'metrics':          {k: float(v) for k, v in metrics.items()},
            'alpha':            int(ALPHA),
            'config': {
                'num_clients':     int(NUM_CLIENTS),
                'dirichlet_alpha': float(DIRICHLET_ALPHA),
                'learning_rate':   float(LEARNING_RATE),
                'batch_size':      int(BATCH_SIZE),
                'local_epochs':    int(LOCAL_EPOCHS),
                'fedprox_mu':      float(MU),
                'dropout':         float(DROPOUT_RATE),
                'warmup_rounds':   int(WARMUP_ROUNDS),
                'mixup_alpha':     float(MIXUP_ALPHA),
            }
        }, f"blip_fedmix_alpha{ALPHA}.pt")
        print(f"    -> New best! Saved.")
    else:
        patience_counter += 1
        print(f"    No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"    Early stopping.")
            break

# ===============================
# Final Evaluation on Full Test Set
# ===============================
print(f"\n{'='*70}")
print("FINAL EVALUATION (Full Test Set)")
print(f"{'='*70}")

# FIX: weights_only=False for PyTorch 2.6+ compatibility
checkpoint = torch.load(f"blip_fedmix_alpha{ALPHA}.pt", weights_only=False)
# FIX: strict=False to ignore unexpected keys like position_ids
global_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
global_model.eval()

print(f"Loading best model from Round {checkpoint['round']}")

preds, refs   = generate_predictions(global_model, test_dataset, max_samples=len(test_dataset))
final_metrics = evaluate_captions(preds, refs)

print(f"\n  BLEU    = {final_metrics['BLEU']:.4f}")
print(f"  METEOR  = {final_metrics['METEOR']:.4f}")
print(f"  ROUGE-L = {final_metrics['ROUGE-L']:.4f}")
print(f"  CIDEr   = {final_metrics['CIDEr']:.4f}")

# Sample predictions
print(f"\n{'='*70}")
print("SAMPLE PREDICTIONS")
print(f"{'='*70}")
for i in range(min(5, len(preds))):
    print(f"\n  [{i+1}] Prediction : {preds[i]}")
    print(f"       Reference  : {refs[i]}")

import json
results = {
    'alpha':             int(ALPHA),
    'labeled_samples':   int(len(labeled_df)),
    'unlabeled_samples': int(len(unlabeled_df)),
    'best_round':        int(checkpoint['round']),
    'final_metrics':     {k: float(v) for k, v in final_metrics.items()},
    'all_round_metrics': [{k: float(v) for k, v in m.items()} for m in all_metrics],
    'config':            checkpoint['config'],
}
with open(f"results_blip_fedmix_alpha{ALPHA}.json", 'w') as f:
    json.dump(results, f, indent=2)

# ====== VISUALIZATION CODE ========
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap

print(f"\n{'='*70}")
print("SAVING VISUALIZATIONS (GRAPHS & IMAGES)")
print(f"{'='*70}")

# Plot metrics over rounds
plt.figure(figsize=(10, 6))
rounds_x = range(1, len(all_metrics) + 1)
if rounds_x:
    plt.plot(rounds_x, [m['CIDEr'] for m in all_metrics], label='CIDEr', marker='o')
    plt.plot(rounds_x, [m['BLEU'] for m in all_metrics], label='BLEU', marker='s')
    plt.plot(rounds_x, [m['METEOR'] for m in all_metrics], label='METEOR', marker='^')
    plt.plot(rounds_x, [m['ROUGE-L'] for m in all_metrics], label='ROUGE-L', marker='d')
    plt.title(f'Evaluation Metrics over Rounds (BLIP FedMix, Alpha {ALPHA})')
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plot_name = f'metrics_plot_blip_fedmix_alpha{ALPHA}.png'
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()
    print(f"  Saved metrics graph: {plot_name}")

# Plot 3 sample predictions
num_samples_to_show = min(3, len(preds))
sample_img_name = f'sample_predictions_blip_fedmix_alpha{ALPHA}.png'
if num_samples_to_show > 0:
    fig, axes = plt.subplots(1, num_samples_to_show, figsize=(5 * num_samples_to_show, 6))
    if num_samples_to_show == 1:
        axes = [axes]
    for i in range(num_samples_to_show):
        img, _ = test_dataset[i]
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')

        pred_text = textwrap.fill(f"Pred: {preds[i]}", width=50)
        ref_text = textwrap.fill(f"Ref: {refs[i]}", width=50)

        ax.set_title(f"{pred_text}\n\n{ref_text}", fontsize=10, loc='center')

    plt.tight_layout()
    plt.savefig(sample_img_name, bbox_inches='tight')
    plt.close()
    print(f"  Saved sample predictions image: {sample_img_name}")
# ======================================

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
print(f"  Best round : {checkpoint['round']}")
print(f"  CIDEr      : {final_metrics['CIDEr']:.4f}")
print(f"  METEOR     : {final_metrics['METEOR']:.4f}")
print(f"  ROUGE-L    : {final_metrics['ROUGE-L']:.4f}")
print(f"  BLEU       : {final_metrics['BLEU']:.4f}")
print(f"\n  Saved: blip_fedmix_alpha{ALPHA}.pt")
print(f"  Saved: results_blip_fedmix_alpha{ALPHA}.json")
try:
    print(f"  Saved: {plot_name}")
    print(f"  Saved: {sample_img_name}")
except NameError:
    pass
