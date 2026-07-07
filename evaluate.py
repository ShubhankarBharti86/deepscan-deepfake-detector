"""
DeepSCN — Evaluation & Benchmarking
Run full evaluation on a test set: AUC, accuracy, F1, confusion matrix,
ROC curve, per-class breakdown, and speed benchmarks.

Usage:
    python evaluate.py --checkpoint checkpoints/deepscn_best.pth --data_root ./data/test
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, average_precision_score,
)
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent))

from models.deepscn_model import build_model
from utils.preprocess import DeepSCNDataset, get_val_transform


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    times = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        t0   = time.perf_counter()
        logits = model(imgs)
        times.append((time.perf_counter() - t0) / imgs.size(0) * 1000)  # ms/image

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.extend(probs[:, 1].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return (
        np.array(all_probs),
        np.array(all_preds),
        np.array(all_labels),
        np.mean(times),
    )


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
DARK_BG = "#0f0f0f"
ACCENT  = "#00e5ff"


def plot_roc_curve(labels, probs, save_path: str):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "w--", lw=0.8, alpha=0.3)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curve — DeepSCN", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG)
    plt.close()
    return auc


def plot_confusion(labels, preds, save_path: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG)
    plt.close()


def plot_score_distribution(labels, probs, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.hist(probs[labels == 0], bins=50, alpha=0.7, label="Real", color="#44ff88")
    ax.hist(probs[labels == 1], bins=50, alpha=0.7, label="Fake", color="#ff4444")
    ax.axvline(0.5, color="white", ls="--", lw=0.8)
    ax.set_xlabel("Fake Probability", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Score Distribution", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=DARK_BG)
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model("full").to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint loaded: epoch {ckpt.get('epoch', '?')}, AUC {ckpt.get('val_auc', '?'):.4f}")

    # Dataset
    ds     = DeepSCNDataset(args.data_root, split="val", extract_faces=args.extract_faces)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test samples: {len(ds)}")

    # Inference
    probs, preds, labels, ms_per_img = run_inference(model, loader, device)

    # Metrics
    auc = plot_roc_curve(labels, probs, str(out_dir / "roc_curve.png"))
    plot_confusion(labels, preds, str(out_dir / "confusion_matrix.png"))
    plot_score_distribution(labels, probs, str(out_dir / "score_dist.png"))

    acc    = (preds == labels).mean() * 100
    ap     = average_precision_score(labels, probs)
    report = classification_report(labels, preds, target_names=["Real", "Fake"])

    summary = {
        "accuracy":          round(acc, 2),
        "roc_auc":           round(auc, 4),
        "avg_precision":     round(ap, 4),
        "inference_ms_per_image": round(ms_per_img, 2),
        "test_samples":      len(ds),
    }

    print("\n" + "=" * 50)
    print("  DeepSCN Evaluation Results")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("\n" + report)

    with open(out_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPlots and summary saved to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="DeepSCN Evaluation")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--data_root",     type=str, required=True)
    p.add_argument("--output_dir",    type=str, default="./eval_outputs")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--extract_faces", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
