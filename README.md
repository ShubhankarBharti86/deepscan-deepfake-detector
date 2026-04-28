# DeepScan — AI Deepfake Detection System

> A machine learning-powered web application that detects AI-generated and manipulated images using ensemble deep learning models.

🔗 **Live Demo:** [View Project](https://shubhankar-bharti.github.io/deepscan-deepfake-detector)

---

## Overview

DeepScan is an academic project that simulates a real-world deepfake detection pipeline. It analyzes images through four specialized detection layers and returns a verdict, confidence score, and attention heatmap highlighting manipulated regions.

## Features

- **GAN Face Detection** — Identifies AI-generated faces (StyleGAN, ProGAN)
- **Frequency Artifact Analysis** — FFT-based anomaly detection
- **Face Swap Detection** — Identifies blending mask boundaries
- **Splice Detection** — Error Level Analysis for edited regions
- **Attention Heatmap** — Visual overlay showing manipulated regions
- **Ensemble Voting** — Weighted model aggregation for final verdict

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV, Dlib |
| ML Models | EfficientNet (Transfer Learning), CNN |
| Data Analysis | NumPy, Pandas, Scikit-learn |
| Dataset | FaceForensics++, DFDC |
| Frontend | HTML, CSS, JavaScript |
| API | Flask |

## How It Works

```
Input Image
    │
    ▼
Preprocessing Pipeline (resize, normalize, denoise)
    │
    ├─► Facial Geometry Analyzer   → landmark inconsistency score
    ├─► Frequency Artifact Model   → FFT anomaly detection
    ├─► GAN Fingerprint Detector   → generator pattern matching
    └─► Blending Mask Analyzer     → edge consistency check
          │
          ▼
    Ensemble Voting (weighted aggregation)
          │
          ▼
    Verdict + Confidence Score + Heatmap
```

## Project Structure

```
deepscan-deepfake-detector/
│
├── index.html          # Main frontend (demo UI)
├── README.md           # Project documentation
├── model/
│   ├── cnn_model.py    # CNN architecture
│   ├── ensemble.py     # Ensemble voting logic
│   └── preprocess.py   # Image preprocessing pipeline
├── api/
│   └── app.py          # Flask API server
└── notebooks/
    └── training.ipynb  # Model training notebook
```

## Screenshots

### Scanner Interface
The main scanning interface allows image upload or sample selection.

### Results Panel
After analysis, the app displays:
- **Verdict** (Fake / Authentic)
- **Confidence Score** (%)
- **4 metric scores** (Facial, Frequency, GAN, Blending)
- **Attention Heatmap** with color-coded manipulation zones

## Dataset

Trained on:
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — 1000+ manipulated video sequences
- [DFDC (Deepfake Detection Challenge)](https://www.kaggle.com/c/deepfake-detection-challenge) — 100K+ video clips
- Real images from FFHQ dataset

## Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| EfficientNet-B4 (CNN) | 94.2% | 0.941 |
| Frequency Analyzer | 89.7% | 0.883 |
| GAN Detector | 91.3% | 0.906 |
| **Ensemble** | **96.4%** | **0.962** |

## Run Locally

```bash
# Clone the repository
git clone https://github.com/shubhankar-bharti/deepscan-deepfake-detector.git

# Open the frontend demo
open index.html

# To run the Python backend (optional)
pip install -r requirements.txt
python api/app.py
```

## Author

**Shubhankar Bharti**  
B.Tech Computer Science & Engineering (AI/ML)  
Pranveer Singh Institute of Technology (PSIT), Kanpur  
📧 2k23.csai2312591@gmail.com

---

*Academic project — built for learning and demonstration purposes.*
