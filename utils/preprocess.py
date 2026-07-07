"""
DeepSCN — Preprocessing Pipeline
Face detection, alignment, augmentation, and dataset loading.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import MTCNN


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]   # ImageNet stats
STD        = [0.229, 0.224, 0.225]
LABEL_MAP  = {"real": 0, "fake": 1}


# ──────────────────────────────────────────────
# Face Detector
# ──────────────────────────────────────────────
class FaceExtractor:
    """
    Wraps MTCNN for robust multi-scale face detection.
    Falls back to center-crop if no face is found.
    """

    def __init__(self, image_size=IMG_SIZE, margin=20, device="cpu"):
        self.image_size = image_size
        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=False,
            device=device,
            post_process=False,   # return raw uint8 pixels
        )

    def extract(self, img: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Args:
            img: RGB image (H×W×3), numpy or PIL
        Returns:
            Face crop (224×224×3) uint8, or None if no face found
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        face = self.detector(img)
        if face is None:
            # Fallback: centre crop
            img_arr = np.array(img)
            h, w = img_arr.shape[:2]
            s = min(h, w)
            y, x = (h - s) // 2, (w - s) // 2
            crop = img_arr[y:y+s, x:x+s]
            crop = cv2.resize(crop, (self.image_size, self.image_size))
            return crop

        # face tensor: (3, H, W) float in [0,255]
        face_np = face.permute(1, 2, 0).byte().numpy()
        return face_np


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
def get_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def denormalize(tensor):
    """Undo ImageNet normalisation for visualisation."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class DeepSCNDataset(Dataset):
    """
    Expects directory layout:
        root/
          real/  *.jpg *.png
          fake/  *.jpg *.png

    Or a flat CSV with columns: path, label (0/1)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        extract_faces: bool = True,
        csv_path: Optional[str] = None,
    ):
        self.transform = transform or (
            get_train_transform() if split == "train" else get_val_transform()
        )
        self.extract_faces = extract_faces
        self.extractor = FaceExtractor() if extract_faces else None
        self.samples: List[Tuple[str, int]] = []

        if csv_path:
            self._load_from_csv(csv_path)
        else:
            self._load_from_dir(root)

    def _load_from_dir(self, root: str):
        root = Path(root)
        for cls_name, label in LABEL_MAP.items():
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for p in cls_dir.glob(ext):
                    self.samples.append((str(p), label))

    def _load_from_csv(self, csv_path: str):
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.extract_faces and self.extractor:
            face = self.extractor.extract(img)
            if face is not None:
                img = face

        img_t = self.transform(img)
        return img_t, label


# ──────────────────────────────────────────────
# DataLoaders factory
# ──────────────────────────────────────────────
def get_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    extract_faces: bool = True,
):
    from sklearn.model_selection import train_test_split

    full_ds = DeepSCNDataset(data_root, split="train", extract_faces=extract_faces)
    indices = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=[full_ds.samples[i][1] for i in indices]
    )

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_full  = DeepSCNDataset(data_root, split="val", extract_faces=extract_faces)
    val_ds    = torch.utils.data.Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────
# Video frame extractor
# ──────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: int = 20, step: int = 10) -> List[np.ndarray]:
    """Sample evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_num = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_num += 1
    cap.release()
    return frames


if __name__ == "__main__":
    extractor = FaceExtractor()
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    face = extractor.extract(dummy_img)
    print(f"Face crop shape: {face.shape if face is not None else 'None'}")

    t = get_train_transform()
    tensor = t(dummy_img)
    print(f"Tensor shape: {tensor.shape}")
