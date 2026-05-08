# Air-Writing Character Recognition

A real-time air-writing recognition system using a **Transformer neural network** trained on pen stroke sequences. Draw characters in the air with your index finger — the model recognizes them live via webcam.

Built for Deep Learning (Spring 2025).

---

## Demo

Draw single or multi-stroke characters in the air. The system predicts what you wrote in real time, with live trajectory forecasting shown ahead of your fingertip.

| Feature | Description |
|---|---|
| Input | Webcam via MediaPipe hand tracking |
| Classes | 97 characters (letters, digits, symbols) |
| Inference | Real-time on CPU or GPU |

---

## ️ Architecture

**BERT-style Transformer Encoder** operating on raw (x, y) stroke coordinate sequences.

- `d_model = 256`, 8 attention heads, 6 encoder layers
- CLS token for classification (pre-norm, dropout = 0.15)
- Positional encoding via learned embeddings
- **5,021,281** trainable parameters

---

##  Results

| Metric | Score |
|---|---|
| Top-1 Accuracy | **71.7%** |
| Top-5 Accuracy | **95.4%** |
| Macro F1 | **0.7125** |

> Confusion clusters on visually similar characters (O/0, l/I/1) are expected and noted.

---

##  Improvements Implemented

### 8.1 — Time Series Forecasting
The forecaster takes the **first 50%** of a stroke and predicts the remaining trajectory as (x, y) coordinates.
- Regression head (MSE loss) instead of classification
- Warm-started from classifier encoder weights
- Shown as **fading blue dots** ahead of your fingertip in the live demo
- Best val MSE: **0.054**

### 8.2 — Multi-Stroke Support
Characters like `i`, `j`, `t`, `f` require multiple strokes. These are joined with a **separator token `[-1, -1]`** between strokes, which the Transformer treats as a stroke boundary marker.

### 8.3 — Writer-Adaptive Fine-Tuning (LoRA)
Low-Rank Adaptation (rank=4, alpha=8) injected into attention output projections and the classifier head.
- Only **~0.3%** of parameters are trained during adaptation
- Fine-tunes on just **5 samples per class**
- Improves accuracy for new writing styles

### 8.4 — Confidence Thresholding
Threshold = **0.50** calibrated on the validation set (accuracy 80.5%). Predictions below threshold are rejected as "uncertain" instead of outputting a wrong guess.

---

## Quick Start

### Requirements
```bash
pip install torch opencv-python mediapipe numpy
```

### Run the Live Demo
```bash
python demo_final.py
```

Make sure these files are in the same directory:
- `best_model.pt` (or `lora_model.pt` for writer-adaptive)
- `label_encoder.pkl`
- `hand_landmarker.task`
- `forecaster.pt`

### Controls

| Key | Action |
|---|---|
| _(draw)_ | Point index finger and write |
| `SPACE` | Save current stroke (multi-stroke mode) |
| `ENTER` | Predict from all saved strokes |
| `C` | Clear and start over |
| `Q` | Quit |

---

##  Repository Structure

```
├── Air_Writing_DL.ipynb     # Full training pipeline (Colab)
├── demo_final.py            # Live webcam demo
├── best_model.pt            # Trained classifier weights
├── forecaster.pt            # Trajectory forecaster weights
├── lora_model.pt            # Writer-adapted model weights
├── label_encoder.pkl        # sklearn LabelEncoder (97 classes)
├── hand_landmarker.task     # MediaPipe hand landmark model
└── Air_Writing_DL.pptx      # Project presentation
```



---

## Training (Colab)

1. Open `Air_Writing_DL.ipynb` in Google Colab
2. Upload `ujipenchars2.txt` (UJI Pen Characters 2 dataset)
3. Set Runtime → **T4 GPU**
4. Run All

Dataset: [UJI Pen Characters 2](https://archive.ics.uci.edu/dataset/493/uji+pen+characters+v2) — 11,640 samples, 97 classes, stratified 70/15/15 split.

---

## Dataset

This project uses the **UJI Pen Characters 2** dataset (UNIPEN format). The dataset file `ujipenchars2.txt` is **not included** in this repo. Download it from [UCI ML Repository](https://archive.ics.uci.edu/dataset/493/uji+pen+characters+v2).
