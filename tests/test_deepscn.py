"""
DeepSCN — Unit Tests
Run: pytest tests/ -v
"""

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deepscn_model import build_model, DeepSCN, DeepSCN_Lite, CBAM, FrequencyBranch
from utils.preprocess import get_train_transform, get_val_transform, denormalize
from models.ensemble import lbp_features, noise_residual_features


# ──────────────────────────────────────────────
# Model tests
# ──────────────────────────────────────────────

class TestDeepSCNModel:
    def test_build_full(self):
        model = build_model("full", pretrained=False)
        assert isinstance(model, DeepSCN)

    def test_build_lite(self):
        model = build_model("lite")
        assert isinstance(model, DeepSCN_Lite)

    def test_forward_shape(self):
        model = build_model("full", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"

    def test_predict_proba_sums_to_one(self):
        model = build_model("full", pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_cbam_output_shape(self):
        cbam = CBAM(channels=64)
        x = torch.randn(2, 64, 14, 14)
        out = cbam(x)
        assert out.shape == x.shape

    def test_frequency_branch(self):
        branch = FrequencyBranch(out_features=256)
        x = torch.randn(2, 3, 224, 224)
        out = branch(x)
        assert out.shape == (2, 256)

    def test_lite_forward(self):
        model = build_model("lite")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)


# ──────────────────────────────────────────────
# Transform tests
# ──────────────────────────────────────────────

class TestTransforms:
    def test_train_transform_output(self):
        t = get_train_transform()
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = t(img)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_val_transform_deterministic(self):
        t = get_val_transform()
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        t1 = t(img)
        t2 = t(img)
        assert torch.allclose(t1, t2)

    def test_denormalize_range(self):
        t = get_val_transform()
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        tensor = t(img)
        out = denormalize(tensor)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5


# ──────────────────────────────────────────────
# Forensic feature tests
# ──────────────────────────────────────────────

class TestForensicFeatures:
    def test_lbp_shape(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feats = lbp_features(img)
        assert feats.shape == (64,)

    def test_noise_shape(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feats = noise_residual_features(img)
        assert feats.shape == (5,)

    def test_lbp_normalized(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feats = lbp_features(img)
        assert abs(feats.sum() - 1.0) < 0.1, "LBP histogram should be approximately normalized"


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
