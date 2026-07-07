"""
DeepSCN - Deep Scene Cognition Network
Core deepfake detection model using EfficientNet-B4 + custom attention head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet


# ──────────────────────────────────────────────
# Spatial Attention Module
# ──────────────────────────────────────────────
class SpatialAttention(nn.Module):
    """
    Learns where to look in the face — highlights
    blending boundaries, texture inconsistencies, etc.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, mx], dim=1)
        return x * self.sigmoid(self.conv(cat))


# ──────────────────────────────────────────────
# Channel Attention (SE block)
# ──────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


# ──────────────────────────────────────────────
# CBAM — Convolutional Block Attention Module
# ──────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ──────────────────────────────────────────────
# Frequency Analysis Branch (DCT / FFT artifacts)
# ──────────────────────────────────────────────
class FrequencyBranch(nn.Module):
    """
    Detects GAN-induced frequency artifacts invisible to the naked eye.
    """
    def __init__(self, out_features=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_features)

    def forward(self, x):
        # Convert to frequency domain via FFT magnitude
        freq = torch.fft.fft2(x.float())
        freq = torch.abs(freq)
        freq = torch.log1p(freq)
        # Normalize to [0,1]
        freq_min = freq.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        freq_max = freq.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        freq = (freq - freq_min) / (freq_max - freq_min + 1e-8)

        out = self.conv(freq)
        out = out.view(out.size(0), -1)
        return F.relu(self.fc(out))


# ──────────────────────────────────────────────
# DeepSCN — Main Model
# ──────────────────────────────────────────────
class DeepSCN(nn.Module):
    """
    DeepSCN: Deep Scene Cognition Network for Deepfake Detection.

    Architecture:
      ┌─────────────────────────────────────────────┐
      │  Input image (224×224 RGB)                  │
      │         │                │                  │
      │  EfficientNet-B4     FrequencyBranch        │
      │  (spatial features)  (FFT artifacts)        │
      │         │                │                  │
      │        CBAM            fc(256)               │
      │         │                │                  │
      │      GAP(1792)           │                  │
      │         └────────────────┘                  │
      │              Concat(2048)                   │
      │                  │                          │
      │           FC → BN → Dropout                 │
      │                  │                          │
      │            Logit (real/fake)                │
      └─────────────────────────────────────────────┘
    """

    def __init__(self, num_classes=2, dropout=0.4, pretrained=True):
        super().__init__()

        # ── Backbone
        self.backbone = EfficientNet.from_pretrained("efficientnet-b4") if pretrained \
                        else EfficientNet.from_name("efficientnet-b4")

        backbone_features = 1792   # EfficientNet-B4 output channels
        self.cbam = CBAM(backbone_features)

        # ── Frequency branch
        self.freq_branch = FrequencyBranch(out_features=256)

        # ── Fusion head
        fusion_in = backbone_features + 256
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Spatial path
        feat = self.backbone.extract_features(x)   # (B, 1792, H, W)
        feat = self.cbam(feat)
        feat = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)

        # Frequency path
        freq_feat = self.freq_branch(x)

        # Fuse & classify
        fused = torch.cat([feat, freq_feat], dim=1)
        return self.head(fused)

    def predict_proba(self, x):
        """Returns softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# ──────────────────────────────────────────────
# Lightweight variant (for edge / CPU inference)
# ──────────────────────────────────────────────
class DeepSCN_Lite(nn.Module):
    """
    MobileNetV3-Small backbone for fast CPU inference.
    Trade ~3% accuracy for 10× speed.
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.backbone(x)).view(x.size(0), -1)
        return self.head(x)


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────
def build_model(variant="full", **kwargs):
    if variant == "full":
        return DeepSCN(**kwargs)
    elif variant == "lite":
        return DeepSCN_Lite(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose 'full' or 'lite'.")


if __name__ == "__main__":
    model = build_model("full")
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # (2, 2)
    probs = model.predict_proba(dummy)
    print(f"Fake probability: {probs[:, 1].tolist()}")
