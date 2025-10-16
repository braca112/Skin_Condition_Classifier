# Skin Condition Classifier (HAM10000) — Demo

**Short description:**  
This repository contains a demo skin lesion classifier trained on the HAM10000 dataset (Skin Cancer MNIST). The project uses transfer learning (MobileNetV2) and provides a demo Streamlit web app for uploading images and getting model predictions.

---

## What is HAM10000?
Download dataset to try.
The HAM10000 ("Human Against Machine") dataset is a multi-source collection of **~10,015** dermatoscopic images across 7 diagnostic categories. It is widely used for research and benchmarking in skin lesion classification. (See dataset on Kaggle). :contentReference[oaicite:4]{index=4}

---

## Structure
- `train.py` — training script (TensorFlow + MobileNetV2).  
- `app.py` — Streamlit app (upload image → prediction).  
- `models/skin_model.h5` — place your trained model here (gitignored).  
- `data/HAM10000/` — place dataset here after download (images + metadata csv).  
- `requirements.txt` — Python dependencies.

---

## How to download HAM10000 (recommended)
1. Create a free Kaggle account and generate an API token:
   - Go to your Kaggle profile → **Account** → **Create New API Token**. This downloads a `kaggle.json` file. :contentReference[oaicite:5]{index=5}

2. Install `kaggle` CLI:
```bash
pip install kaggle
