"""
DeepSCN — Ensemble Detector
Combines deep CNN predictions with handcrafted forensic features:
  • ELA  (Error Level Analysis)
  • LBP  (Local Binary Pattern texture)
  • Noise residuals
  • XGBoost meta-classifier
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Union

from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

from models.deepscn_model import build_model
from utils.preprocess import get_val_transform, FaceExtractor


# ──────────────────────────────────────────────
# Forensic Feature Extractors
# ──────────────────────────────────────────────

def ela_features(img_path: str, quality: int = 90, resize: tuple = (128, 128)) -> np.ndarray:
    """
    Error Level Analysis: re-saves JPEG at lower quality and measures
    per-pixel difference. Deepfakes show distinctive ELA patterns.
    """
    orig = Image.open(img_path).convert("RGB")
    orig = orig.resize(resize)

    from io import BytesIO
    buf = BytesIO()
    orig.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    ela_arr = np.array(orig, dtype=np.float32) - np.array(recompressed, dtype=np.float32)
    ela_arr = np.abs(ela_arr)

    # Statistical summary per channel
    feats = []
    for c in range(3):
        ch = ela_arr[:, :, c]
        feats += [ch.mean(), ch.std(), ch.max(), np.percentile(ch, 75)]
    return np.array(feats, dtype=np.float32)   # 12-dim


def lbp_features(img: np.ndarray, P: int = 8, R: float = 1.0, n_bins: int = 64) -> np.ndarray:
    """
    Local Binary Pattern histogram. GAN-generated faces have subtly
    different micro-texture statistics to real faces.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)   # 64-dim


def noise_residual_features(img: np.ndarray) -> np.ndarray:
    """
    Noise residual statistics. Splicing / GAN artifacts leave traces
    in high-frequency noise components.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Wavelet-like: original minus Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray - blurred

    feats = [
        residual.mean(),
        residual.std(),
        np.percentile(residual, 25),
        np.percentile(residual, 75),
        (residual > residual.std()).mean(),    # fraction of "high noise" pixels
    ]
    return np.array(feats, dtype=np.float32)   # 5-dim


def extract_handcrafted(img_path: str, img_rgb: np.ndarray) -> np.ndarray:
    """Concatenate all forensic features → 81-dim vector."""
    ela  = ela_features(img_path)
    lbp  = lbp_features(img_rgb)
    noise = noise_residual_features(img_rgb)
    return np.concatenate([ela, lbp, noise])


# ──────────────────────────────────────────────
# Deep Feature Extractor (penultimate layer)
# ──────────────────────────────────────────────

class DeepFeatureExtractor:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = build_model("full").to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.transform    = get_val_transform()
        self.face_extractor = FaceExtractor(device=device)

        # Hook to grab penultimate (128-dim) features
        self._features: List[torch.Tensor] = []
        self.model.head[-1].register_forward_hook(
            lambda m, i, o: self._features.append(i[0].detach())
        )

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        self._features.clear()
        face = self.face_extractor.extract(img_rgb) or img_rgb
        tensor = self.transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(tensor)
        return self._features[0].squeeze(0).cpu().numpy()   # (128,)


# ──────────────────────────────────────────────
# Ensemble Model
# ──────────────────────────────────────────────

class DeepSCNEnsemble:
    """
    Three-stage ensemble:
      1. DeepSCN CNN → softmax prob
      2. Handcrafted forensics → XGBoost prob
      3. Weighted average fusion
    """

    def __init__(
        self,
        cnn_checkpoint: str,
        xgb_model_path: str = None,
        cnn_weight: float = 0.7,
        device: str = "cpu",
    ):
        self.device     = device
        self.cnn_weight = cnn_weight
        self.xgb_weight = 1.0 - cnn_weight

        # CNN
        self.cnn_model = build_model("full").to(torch.device(device))
        ckpt = torch.load(cnn_checkpoint, map_location=device)
        self.cnn_model.load_state_dict(ckpt["model_state_dict"])
        self.cnn_model.eval()

        self.transform     = get_val_transform()
        self.face_extractor = FaceExtractor(device=device)

        # XGBoost (optional, loaded if exists)
        self.xgb_clf = None
        if xgb_model_path and Path(xgb_model_path).exists():
            self.xgb_clf = joblib.load(xgb_model_path)

    # ── CNN prediction
    def _cnn_predict(self, img_rgb: np.ndarray) -> float:
        face = self.face_extractor.extract(img_rgb) or img_rgb
        t = self.transform(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = F.softmax(self.cnn_model(t), dim=1)[0, 1].item()
        return prob

    # ── XGBoost prediction
    def _xgb_predict(self, img_path: str, img_rgb: np.ndarray) -> float:
        if self.xgb_clf is None:
            return 0.5   # neutral if not trained yet
        feats = extract_handcrafted(img_path, img_rgb).reshape(1, -1)
        return self.xgb_clf.predict_proba(feats)[0, 1]

    # ── Ensemble prediction
    def predict(self, img_path: str) -> dict:
        import cv2
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cnn_prob = self._cnn_predict(img_rgb)
        xgb_prob = self._xgb_predict(img_path, img_rgb)

        ensemble_prob = self.cnn_weight * cnn_prob + self.xgb_weight * xgb_prob
        label = "FAKE" if ensemble_prob > 0.5 else "REAL"
        confidence = ensemble_prob if label == "FAKE" else 1 - ensemble_prob

        return {
            "label":          label,
            "confidence":     round(confidence * 100, 2),
            "fake_prob":      round(ensemble_prob * 100, 2),
            "cnn_prob":       round(cnn_prob * 100, 2),
            "xgb_prob":       round(xgb_prob * 100, 2),
        }

    # ── Train XGBoost on extracted features
    def train_xgb(self, img_paths: List[str], labels: List[int], save_path: str = "checkpoints/xgb.pkl"):
        print("Extracting forensic features...")
        X, y = [], []
        for path, lbl in zip(img_paths, labels):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            X.append(extract_handcrafted(path, img))
            y.append(lbl)

        X = np.array(X)
        y = np.array(y)

        self.xgb_clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.xgb_clf.fit(X, y, eval_set=[(X, y)], verbose=100)
        joblib.dump(self.xgb_clf, save_path)
        print(f"XGBoost saved to {save_path}")


if __name__ == "__main__":
    # Demo usage (no checkpoint needed — shows feature shapes)
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    lbp = lbp_features(dummy)
    noise = noise_residual_features(dummy)
    print(f"LBP features:   {lbp.shape}")
    print(f"Noise features: {noise.shape}")
