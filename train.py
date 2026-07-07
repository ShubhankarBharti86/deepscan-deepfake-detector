"""
DeepSCN — Training Script
Full training loop: mixed precision, early stopping, LR scheduling,
Grad-CAM checkpointing, and rich console logging.

Usage:
    python train.py --data_root ./data --epochs 30 --batch_size 32
"""

import os
import time
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, classification_report

import sys
sys.path.append(str(Path(__file__).parent))

from models.deepscn_model import build_model
from utils.preprocess import get_loaders

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DeepSCN")


# ──────────────────────────────────────────────
# Focal Loss (handles class imbalance)
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.correct    = 0
        self.total      = 0
        self.all_probs  = []
        self.all_labels = []

    def update(self, loss, preds, probs, labels):
        self.total_loss += loss
        self.correct    += (preds == labels).sum().item()
        self.total      += labels.size(0)
        self.all_probs.extend(probs[:, 1].cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())

    @property
    def avg_loss(self):
        return self.total_loss / max(self.total, 1)

    @property
    def accuracy(self):
        return 100.0 * self.correct / max(self.total, 1)

    @property
    def auc(self):
        try:
            return roc_auc_score(self.all_labels, self.all_probs)
        except Exception:
            return 0.0


# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=7, delta=1e-4):
        self.patience  = patience
        self.delta     = delta
        self.best_val  = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss):
        if val_loss < self.best_val - self.delta:
            self.best_val = val_loss
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ──────────────────────────────────────────────
# Train / Val Epoch
# ──────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device, training=True):
    model.train() if training else model.eval()
    tracker = MetricTracker()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            tracker.update(loss.item(), preds, probs, labels)

        if batch_idx % 50 == 0:
            log.info(
                f"  {'Train' if training else 'Val'} step {batch_idx}/{len(loader)} "
                f"| loss={loss.item():.4f} | acc={tracker.accuracy:.1f}%"
            )

    return tracker


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data
    train_loader, val_loader = get_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        extract_faces=args.extract_faces,
    )
    log.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model
    model = build_model(variant=args.variant).to(device)
    log.info(f"Model: DeepSCN-{args.variant} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = GradScaler()
    stopper   = EarlyStopping(patience=args.patience)

    best_auc   = 0.0
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    ckpt_path  = Path(args.output_dir) / "deepscn_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Starting DeepSCN training")
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_t = run_epoch(model, train_loader, criterion, optimizer, scaler, device, training=True)
        val_t   = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, training=False)

        scheduler.step(epoch)

        history["train_loss"].append(train_t.avg_loss)
        history["val_loss"].append(val_t.avg_loss)
        history["train_acc"].append(train_t.accuracy)
        history["val_acc"].append(val_t.accuracy)
        history["val_auc"].append(val_t.auc)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch:03d}/{args.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_t.avg_loss:.4f} acc={train_t.accuracy:.1f}% | "
            f"val_loss={val_t.avg_loss:.4f} acc={val_t.accuracy:.1f}% AUC={val_t.auc:.4f}"
        )

        # Save best
        if val_t.auc > best_auc:
            best_auc = val_t.auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc,
                "args": vars(args),
            }, ckpt_path)
            log.info(f"  ✓ New best AUC={best_auc:.4f} → saved to {ckpt_path}")

        if stopper.step(val_t.avg_loss):
            log.info(f"Early stopping at epoch {epoch}.")
            break

    # Save history
    hist_path = Path(args.output_dir) / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved to {hist_path}")
    log.info(f"Best Validation AUC: {best_auc:.4f}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeepSCN Deepfake Detection Trainer")
    p.add_argument("--data_root",    type=str, default="./data",      help="Root data directory")
    p.add_argument("--output_dir",   type=str, default="./checkpoints")
    p.add_argument("--variant",      type=str, default="full",         choices=["full", "lite"])
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--val_split",    type=float, default=0.2)
    p.add_argument("--patience",     type=int, default=7)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--extract_faces", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
