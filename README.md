# DeepSCN / DeepScan — Deepfake Detection

DeepSCN is a PyTorch deepfake detection project using EfficientNet, attention, FFT-frequency features, Grad-CAM explainability, a Flask API, video analysis, and a cyberpunk web UI.

> Important: predictions are only meaningful after training on a real dataset and loading `checkpoints/deepscn_best.pth`. Without a checkpoint, the API starts in demo/random-weight mode.

## Features

- Image deepfake detection
- Video frame sampling endpoint
- Webcam inference script
- EfficientNet-B4 + CBAM attention
- Frequency branch using FFT artifacts
- Focal loss for class imbalance
- Mixed precision training
- AUC, accuracy, classification metrics
- Grad-CAM explanation endpoint
- Flask backend + HTML frontend
- Docker support

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Dataset format

```text
data/
  real/
    real_001.jpg
  fake/
    fake_001.jpg
```

## Train

```bash
python train.py --data_root data --epochs 30 --batch_size 16 --output_dir checkpoints
```

The best checkpoint is saved to:

```text
checkpoints/deepscn_best.pth
```

## Predict one image

```bash
python predict.py --image path/to/image.jpg --checkpoint checkpoints/deepscn_best.pth
```

## Run web app + API

```bash
python api/app.py
```

Open:

```text
http://127.0.0.1:5000
```

Upload an image. The frontend calls `/predict` and displays the backend result.

## Video API

```bash
curl -X POST -F "file=@sample.mp4" http://127.0.0.1:5000/predict/video
```

## Webcam

```bash
python webcam.py --checkpoint checkpoints/deepscn_best.pth
```

Press `q` to quit.

## Docker

```bash
docker compose up --build
```

## Recommended datasets

Use datasets only according to their license/terms:

- FaceForensics++
- DFDC
- Celeb-DF v2
- DeepFakeDetection Dataset

## GitHub notes

Do commit:

- source code
- README
- screenshots
- small sample images if allowed

Do not commit:

- full datasets
- huge checkpoints
- private API keys

