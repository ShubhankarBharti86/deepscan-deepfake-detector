"""
DeepSCN — Grad-CAM Explainability
Produces saliency heatmaps showing which face regions triggered the fake prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from models.deepscn_model import build_model
from utils.preprocess import get_val_transform, FaceExtractor, denormalize


# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # GAP over spatial dims
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ──────────────────────────────────────────────
# Visualiser
# ──────────────────────────────────────────────
class ExplainabilityVisualizer:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = build_model("full").to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Target the last convolutional block of EfficientNet backbone
        target_layer = self.model.backbone._blocks[-1]
        self.gradcam = GradCAM(self.model, target_layer)

        self.transform     = get_val_transform()
        self.face_extractor = FaceExtractor(device=device)

    def explain(self, img_path: str, save_path: Optional[str] = None) -> dict:
        import cv2
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        face = self.face_extractor.extract(img) or img

        tensor = self.transform(face).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        # Forward + Grad-CAM
        cam = self.gradcam.generate(tensor, class_idx=1)   # class 1 = fake

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        fake_prob = probs[1].item()
        label     = "FAKE" if fake_prob > 0.5 else "REAL"

        # Overlay heatmap
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay     = (0.5 * face + 0.5 * heatmap_rgb).astype(np.uint8)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor("#0d0d0d")
        for ax in axes:
            ax.set_facecolor("#0d0d0d")
            ax.axis("off")

        axes[0].imshow(face);           axes[0].set_title("Input Face",  color="white", fontsize=12)
        axes[1].imshow(heatmap_rgb);    axes[1].set_title("Grad-CAM",    color="white", fontsize=12)
        axes[2].imshow(overlay);        axes[2].set_title(
            f"Prediction: {label}\nFake prob: {fake_prob:.1%}",
            color="#ff4444" if label == "FAKE" else "#44ff88",
            fontsize=12,
        )
        plt.tight_layout()

        out_path = save_path or f"gradcam_{Path(img_path).stem}.png"
        plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

        return {
            "label":     label,
            "fake_prob": round(fake_prob * 100, 2),
            "cam_path":  out_path,
        }


# ──────────────────────────────────────────────
# Batch explain
# ──────────────────────────────────────────────
def batch_explain(img_paths, checkpoint_path, out_dir="./gradcam_outputs", device="cpu"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    viz = ExplainabilityVisualizer(checkpoint_path, device)
    results = []
    for p in img_paths:
        save = str(Path(out_dir) / f"{Path(p).stem}_cam.png")
        r = viz.explain(p, save_path=save)
        results.append(r)
        print(f"{Path(p).name}  → {r['label']} ({r['fake_prob']}%)")
    return results


if __name__ == "__main__":
    # Demo: explain a single image (requires checkpoint)
    print("Grad-CAM module ready.")
    print("Usage:  from utils.gradcam import ExplainabilityVisualizer")
    print("        viz = ExplainabilityVisualizer('checkpoints/deepscn_best.pth')")
    print("        result = viz.explain('path/to/face.jpg')")
