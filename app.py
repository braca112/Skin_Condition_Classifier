# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from PIL import Image

MODEL_PATH = "models/skin_model.h5"
IMG_SIZE = (224, 224)

# human-readable labels - must match training mapping
LABELS = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic_nevus",
    "vascular_lesion"
]

@st.cache_resource
def load_model_cached(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error("Model file not found. Please train the model first and place it in models/skin_model.h5")
        return None
    model = load_model(path)
    return model

def preprocess_image(image: Image.Image, target_size=IMG_SIZE):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = img_to_array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

st.set_page_config(page_title="Skin Condition Classifier", layout="centered")
st.title("Skin Condition Classifier — Demo")
st.markdown("""
Upload a dermatoscopic or regular photo of a skin lesion.  
This demo uses a convolutional neural network trained on the HAM10000 dataset. **This is for educational/demonstration purposes only** — not a medical diagnosis.
""")

model = load_model_cached()
if model is None:
    st.stop()

uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    st.write("")
    st.write("Predicting...")
    x = preprocess_image(img)
    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)
    top_prob = preds[top_idx]
    pred_label = model.output_shape[-1]  # not used, we use labels list
    st.markdown(f"**Prediction:** `{LABELS[top_idx]}`")
    st.markdown(f"**Confidence:** `{top_prob*100:.2f}%`")
    # Show class probabilities
    st.subheader("All class probabilities")
    prob_dict = {LABELS[i]: float(preds[i]) for i in range(len(LABELS))}
    st.table({ "class": list(prob_dict.keys()), "probability": [f"{p*100:.2f}%" for p in prob_dict.values()] })
