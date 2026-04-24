# -*- coding: utf-8 -*-
"""
OPTIMIZED SUPERVISED BASELINE - Federated Learning
Goal: Maximize metrics with 100% labeled data within 6-10 hr time budget

Key fixes over previous versions:
- Only decoder unfrozen (stable, fast, proven best)
- Dropout regularization to prevent overfitting
- Higher learning rate with warmup (faster convergence)
- Fewer rounds but more effective local training
- Cosine LR schedule to avoid loss collapse
- FedProx proximal term to reduce client drift
- Larger batch size to stabilize gradients
"""

import os
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import copy, sys, gc
import numpy as np
import pandas as pd
from PIL import Image
import argparse
sys.stdout.reconfigure(line_buffering=True)

import torch
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

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=int, default=100)
args = parser.parse_args()
ALPHA = args.alpha

# ===============================
# TUNED HYPERPARAMETERS
# Key insight: fewer rounds, better each round
# ===============================
NUM_CLIENTS      = 3
DIRICHLET_ALPHA  = 0.5   # More IID (better for baseline)
ROUNDS           = 20      # Enough rounds within time budget
LOCAL_EPOCHS     = 3       # Sweet spot - not too few, not too many
LEARNING_RATE    = 5e-5    # Higher LR for faster convergence
BATCH_SIZE       = 16      # Larger batch = more stable gradients
PATIENCE         = 7       # Give model time to find best
WARMUP_ROUNDS    = 2       # Rounds before LR starts decaying
MU               = 0.01    # FedProx proximal term (reduces client drift)
DROPOUT_RATE     = 0.1     # Regularization

DATASET_DIR = "/home/tanmoyhazra/u23ai048"
CSV_PATH = f"{DATASET_DIR}/indiana_complete.csv"

# ===============================
# Device
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
try:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except:
    print("GPU detected (NVML info unavailable)")

print(f"\n{'='*70}")
print(f"OPTIMIZED SUPERVISED BASELINE FL")
print(f"{'='*70}")
print(f"  Labeled data:    {ALPHA}%")
print(f"  Clients:         {NUM_CLIENTS}")
print(f"  Dirichlet alpha: {DIRICHLET_ALPHA}")
print(f"  Rounds:          {ROUNDS}")
print(f"  Local epochs:    {LOCAL_EPOCHS}")
print(f"  Learning rate:   {LEARNING_RATE}")
print(f"  Batch size:      {BATCH_SIZE}")
print(f"  Patience:        {PATIENCE}")
print(f"  FedProx mu:      {MU}")
print(f"  Dropout:         {DROPOUT_RATE}")

# ===============================
# Dataset
# ===============================
print(f"\n{'='*70}")
print("LOADING DATASET")
print(f"{'='*70}")

df = pd.read_csv(CSV_PATH)
print(f"Total samples: {len(df)}")

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# Select alpha% labeled
np.random.seed(42)
n_labeled = int(len(train_df) * ALPHA / 100.0)
labeled_indices = np.random.permutation(len(train_df))[:n_labeled]
labeled_df = train_df.iloc[labeled_indices].reset_index(drop=True)
print(f"Labeled: {len(labeled_df)} | Unlabeled (ignored): {len(train_df)-len(labeled_df)}")

# ===============================
# Dataset class
# Light augmentation only - heavy augmentation hurt performance
# ===============================
class ChestXrayDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.transform = transforms.Compose([
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
        return img, row["caption"]

train_dataset = ChestXrayDataset(labeled_df, augment=True)
test_dataset  = ChestXrayDataset(test_df,    augment=False)
print(f"Train dataset: {len(train_dataset)} | Test dataset: {len(test_dataset)}")

# ===============================
# Federated Split
# ===============================
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
    # Equal fallback
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
client_splits   = dirichlet_split(len(train_dataset), NUM_CLIENTS, DIRICHLET_ALPHA)
client_datasets = [Subset(train_dataset, s) for s in client_splits]
for i, ds in enumerate(client_datasets):
    print(f"  Client {i}: {len(ds)} samples ({len(ds)/len(train_dataset)*100:.1f}%)")

# ===============================
# Load BLIP - Decoder only unfrozen + dropout
# This is the proven best configuration
# ===============================
print(f"\n{'='*70}")
print("LOADING BLIP MODEL")
print(f"{'='*70}")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

global_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
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
# Collate
# ===============================
def collate_fn(batch):
    imgs, caps = zip(*batch)
    return list(imgs), list(caps)

# ===============================
# Local Training with FedProx
# FedProx adds a proximal term that penalizes client models
# from drifting too far from the global model - key improvement
# ===============================
def local_train(model, dataset, epochs, lr, global_model_ref):
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Cosine LR with warmup - prevents loss collapse
    total_steps = epochs * len(loader)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Get global model parameters for FedProx term
    global_params = {
        name: param.detach().clone()
        for name, param in global_model_ref.named_parameters()
        if param.requires_grad
    }

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for imgs, caps in loader:
            try:
                inputs = processor(
                    images=imgs,
                    text=caps,
                    padding=True,
                    return_tensors="pt"
                ).to(DEVICE)

                outputs = model(**inputs, labels=inputs["input_ids"])
                ce_loss = outputs.loss

                if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                    print(f"      ?? Invalid loss, skipping batch")
                    continue

                # FedProx: add proximal term to penalize client drift
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and name in global_params:
                        prox_term += ((param - global_params[name]) ** 2).sum()
                prox_loss = (MU / 2.0) * prox_term

                loss = ce_loss + prox_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                scheduler.step()

                epoch_loss += ce_loss.item()  # Log only CE loss for clarity
                n_batches += 1

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
# FedAvg
# ===============================
def fedavg(global_model, client_models, client_sizes):
    total = sum(client_sizes)
    gd    = global_model.state_dict()
    for key in gd.keys():
        gd[key] = sum(
            cm.state_dict()[key].cpu() * (s / total)
            for cm, s in zip(client_models, client_sizes)
        )
    global_model.load_state_dict(gd)

# ===============================
# Evaluation
# ===============================
def generate_predictions(model, dataset, max_samples=200):
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
# Training Loop
# Learning rate decays across rounds (outer cosine schedule)
# ===============================
print(f"\n{'='*70}")
print("STARTING OPTIMIZED FEDERATED TRAINING")
print(f"{'='*70}")

BEST_CIDER      = -1
BEST_ROUND      = -1
patience_counter = 0
all_metrics     = []

for round_num in range(ROUNDS):

    # Outer LR schedule: decay learning rate across rounds
    progress  = round_num / max(1, ROUNDS - 1)
    round_lr  = LEARNING_RATE * max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))

    print(f"\n{'='*70}")
    print(f"Round {round_num+1}/{ROUNDS} | LR: {round_lr:.2e} | Best CIDEr so far: {BEST_CIDER:.4f}")
    print(f"{'='*70}")

    client_models = []
    client_sizes  = []

    for cid, cds in enumerate(client_datasets):
        print(f"  Client {cid}: {len(cds)} samples")
        local_model = copy.deepcopy(global_model).to(DEVICE)
        local_train(local_model, cds, LOCAL_EPOCHS, round_lr, global_model)
        client_models.append(local_model)
        client_sizes.append(len(cds))
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  Aggregating (FedAvg)...")
    fedavg(global_model, client_models, client_sizes)
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
        BEST_CIDER = metrics["CIDEr"]
        BEST_ROUND = round_num + 1
        patience_counter = 0
        torch.save({
            'round':            BEST_ROUND,
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
            }
        }, f"baseline_optimized_alpha{ALPHA}.pt")
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

checkpoint = torch.load(f"baseline_optimized_alpha{ALPHA}.pt", weights_only=False)
global_model.load_state_dict(checkpoint['model_state_dict'])
global_model.eval()

print(f"Loading best model from Round {checkpoint['round']}")

preds, refs  = generate_predictions(global_model, test_dataset, max_samples=len(test_dataset))
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
    'alpha':           ALPHA,
    'labeled_samples': len(labeled_df),
    'best_round':      checkpoint['round'],
    'final_metrics':   final_metrics,
    'all_round_metrics': all_metrics,
    'config':          checkpoint['config'],
}
with open(f"results_optimized_alpha{ALPHA}.json", 'w') as f:
    json.dump(results, f, indent=2)

# ====== NEW VISUALIZATION CODE ========
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap

print(f"\n{'='*70}")
print("SAVING VISUALIZATIONS (GRAPHS & IMAGES)")
print(f"{'='*70}")

# Plot metrics
plt.figure(figsize=(10, 6))
rounds = range(1, len(all_metrics) + 1)
if rounds:
    plt.plot(rounds, [m['CIDEr'] for m in all_metrics], label='CIDEr', marker='o')
    plt.plot(rounds, [m['BLEU'] for m in all_metrics], label='BLEU', marker='s')
    plt.plot(rounds, [m['METEOR'] for m in all_metrics], label='METEOR', marker='^')
    plt.plot(rounds, [m['ROUGE-L'] for m in all_metrics], label='ROUGE-L', marker='d')
    plt.title(f'Evaluation Metrics over Rounds (BLIP, Alpha {ALPHA})')
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plot_name = f'metrics_plot_blip_alpha{ALPHA}.png'
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()
    print(f"  Saved metrics graph: {plot_name}")

# Plot 3 sample predictions
num_samples_to_show = min(3, len(preds))
sample_img_name = f'sample_predictions_blip_alpha{ALPHA}.png'
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
print(f"\n  Saved: baseline_optimized_alpha{ALPHA}.pt")
print(f"  Saved: results_optimized_alpha{ALPHA}.json")
try:
    print(f"  Saved: {plot_name}")
    print(f"  Saved: {sample_img_name}")
except NameError:
    pass