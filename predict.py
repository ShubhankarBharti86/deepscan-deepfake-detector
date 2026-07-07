"""
DeepSCN — Inference Script
Run predictions on a single image, a folder, or a video.

Usage:
    python predict.py --input face.jpg
    python predict.py --input ./test_images/
    python predict.py --input clip.mp4 --explain
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))

from models.deepscn_model import build_model
from utils.preprocess import get_val_transform, FaceExtractor, extract_frames

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(checkpoint_path, device):
    model = build_model("full").to(device)
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[✓] Checkpoint loaded ({checkpoint_path})")
    else:
        print(f"[!] No checkpoint at {checkpoint_path}. Using random weights (demo).")
    model.eval()
    return model


@torch.no_grad()
def predict_single(img_rgb, model, transform, extractor, device):
    face = extractor.extract(img_rgb) or img_rgb
    t = transform(face).unsqueeze(0).to(device)
    probs = F.softmax(model(t), dim=1)[0]
    fake_p = probs[1].item()
    label  = "FAKE" if fake_p > 0.5 else "REAL"
    return label, round(fake_p * 100, 1)


def print_result(name, label, fake_p):
    icon  = "🔴" if label == "FAKE" else "🟢"
    bar_w = int(fake_p / 5)
    bar   = "█" * bar_w + "░" * (20 - bar_w)
    conf  = fake_p if label == "FAKE" else 100 - fake_p
    print(f"{icon}  {name:<40} {label}  [{bar}] {conf:.1f}%")


def run(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = load_model(args.checkpoint, device)
    transform = get_val_transform()
    extractor = FaceExtractor(device=str(device))

    inp = Path(args.input)

    # ── Single image
    if inp.suffix.lower() in IMG_EXTS:
        img = cv2.cvtColor(cv2.imread(str(inp)), cv2.COLOR_BGR2RGB)
        label, fake_p = predict_single(img, model, transform, extractor, device)
        print("\n DeepSCN Prediction")
        print(" " + "─" * 60)
        print_result(inp.name, label, fake_p)
        if args.explain:
            from utils.gradcam import ExplainabilityVisualizer
            if Path(args.checkpoint).exists():
                viz = ExplainabilityVisualizer(args.checkpoint, device=str(device))
                r = viz.explain(str(inp))
                print(f"\n[✓] Grad-CAM saved to {r['cam_path']}")

    # ── Folder
    elif inp.is_dir():
        files = [p for p in sorted(inp.iterdir()) if p.suffix.lower() in IMG_EXTS]
        print(f"\n DeepSCN Batch Prediction ({len(files)} images)")
        print(" " + "─" * 60)
        fakes = 0
        for p in files:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            label, fake_p = predict_single(img, model, transform, extractor, device)
            print_result(p.name, label, fake_p)
            if label == "FAKE":
                fakes += 1
        print(" " + "─" * 60)
        print(f" Summary: {fakes}/{len(files)} flagged as FAKE ({fakes/max(len(files),1)*100:.1f}%)\n")

    # ── Video
    elif inp.suffix.lower() in VID_EXTS:
        frames = extract_frames(str(inp), max_frames=30, step=5)
        print(f"\n DeepSCN Video Analysis ({len(frames)} frames sampled)")
        print(" " + "─" * 60)
        probs = []
        for i, frame in enumerate(frames):
            label, fake_p = predict_single(frame, model, transform, extractor, device)
            print_result(f"Frame {i*5:04d}", label, fake_p)
            probs.append(fake_p)
        avg = np.mean(probs)
        final = "FAKE" if avg > 50 else "REAL"
        print(" " + "─" * 60)
        print(f" Video verdict: {final}  (avg fake prob {avg:.1f}%)\n")

    else:
        print(f"Unsupported input: {inp}")
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DeepSCN Inference")
    p.add_argument("--input",      type=str, required=True,  help="Image, folder, or video path")
    p.add_argument("--checkpoint", type=str, default="checkpoints/deepscn_best.pth")
    p.add_argument("--explain",    action="store_true",       help="Generate Grad-CAM explanation")
    run(p.parse_args())
