"""
DeepSCN — REST API
Serves deepfake detection predictions via Flask.

Endpoints:
  POST /predict        — image file upload → label + confidence
  POST /predict/url    — public image URL → label + confidence
  POST /predict/video  — video file → frame-by-frame analysis
  GET  /health         — liveness probe
  GET  /model/info     — model metadata
"""

import os
import sys
import io
import base64
import tempfile
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

from models.deepscn_model import build_model
from utils.preprocess import get_val_transform, FaceExtractor
from utils.gradcam import ExplainabilityVisualizer

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CHECKPOINT   = os.environ.get("DEEPSCN_CKPT", "checkpoints/deepscn_best.pth")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FILE_MB  = 20
ALLOWED_IMG  = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_VID  = {"mp4", "avi", "mov", "mkv"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("DeepSCN-API")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024


# ──────────────────────────────────────────────
# Model loader (singleton)
# ──────────────────────────────────────────────
_model    = None
_transform = get_val_transform()
_extractor = FaceExtractor(device=DEVICE)
_explainer: Optional[ExplainabilityVisualizer] = None


def get_model():
    global _model
    if _model is None:
        _model = build_model("full").to(DEVICE)
        if Path(CHECKPOINT).exists():
            ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
            _model.load_state_dict(ckpt["model_state_dict"])
            log.info(f"Checkpoint loaded from {CHECKPOINT}")
        else:
            log.warning("No checkpoint found — using random weights (demo mode).")
        _model.eval()
    return _model


def get_explainer():
    global _explainer
    if _explainer is None and Path(CHECKPOINT).exists():
        _explainer = ExplainabilityVisualizer(CHECKPOINT, device=DEVICE)
    return _explainer


# ──────────────────────────────────────────────
# Core prediction helper
# ──────────────────────────────────────────────
def predict_image(img_rgb: np.ndarray) -> dict:
    model = get_model()
    face  = _extractor.extract(img_rgb)
    if face is None:
        face = img_rgb
    tensor = _transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    fake_prob = probs[1].item()
    real_prob = probs[0].item()
    label     = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if label == "FAKE" else real_prob

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "fake_prob":  round(fake_prob * 100, 2),
        "real_prob":  round(real_prob * 100, 2),
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": DEVICE, "checkpoint": CHECKPOINT})


@app.route("/model/info", methods=["GET"])
def model_info():
    model = get_model()
    n_params = sum(p.numel() for p in model.parameters())
    return jsonify({
        "architecture": "DeepSCN (EfficientNet-B4 + FrequencyBranch + CBAM)",
        "parameters": n_params,
        "input_size": "224×224",
        "classes": ["real", "fake"],
        "device": DEVICE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use key 'file'."}), 400

    f   = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMG:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image."}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = predict_image(img_rgb)
    result["filename"] = secure_filename(f.filename)
    return jsonify(result)


@app.route("/predict/url", methods=["POST"])
def predict_from_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Provide JSON with key 'url'."}), 400

    import urllib.request
    try:
        with urllib.request.urlopen(data["url"], timeout=10) as resp:
            raw = np.frombuffer(resp.read(), dtype=np.uint8)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image from URL."}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = predict_image(img_rgb)
    result["url"] = data["url"]
    return jsonify(result)


@app.route("/predict/video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    f   = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_VID:
        return jsonify({"error": f"Unsupported video format: {ext}"}), 400

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step  = max(1, total // 20)   # sample up to 20 frames

        frame_results = []
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                r   = predict_image(rgb)
                frame_results.append(r)
            frame_num += 1
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frame_results:
        return jsonify({"error": "Could not extract frames."}), 400

    avg_fake = np.mean([r["fake_prob"] for r in frame_results])
    label    = "FAKE" if avg_fake > 50 else "REAL"

    return jsonify({
        "label":           label,
        "avg_fake_prob":   round(avg_fake, 2),
        "frames_analyzed": len(frame_results),
        "frame_results":   frame_results,
    })


@app.route("/explain", methods=["POST"])
def explain():
    """Return Grad-CAM heatmap overlay as base64 PNG."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    explainer = get_explainer()
    if explainer is None:
        return jsonify({"error": "No checkpoint available for explanation."}), 503

    f = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    out_path = tmp_path.replace(".jpg", "_cam.png")
    try:
        result = explainer.explain(tmp_path, save_path=out_path)
        with open(out_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode()
        result["heatmap_b64"] = b64
    finally:
        for p in [tmp_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)

    return jsonify(result)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Warm up model on startup
    get_model()
    log.info("DeepSCN API ready on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
