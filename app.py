"""
Streamlit app for brain tumour detection.

Run:
streamlit run app.py
"""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


IMG_SIZE = (224, 224)
MODEL_PATH = Path("model/bestmodel.h5")


st.set_page_config(page_title="Brain Tumour Detection", page_icon="🧠")

st.title("Brain Tumour Detection using CNN")
st.write("Upload a brain MRI image to predict whether it contains a tumour.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Train the model first and place `bestmodel.h5` inside the `model/` folder.")
else:
    model = load_model(MODEL_PATH)

    uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            img = load_img(temp_file.name, target_size=IMG_SIZE)

        input_arr = img_to_array(img) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)

        prediction = float(model.predict(input_arr)[0][0])

        st.write(f"Prediction score: `{prediction:.4f}`")

        if prediction >= 0.5:
            st.error("Tumour detected")
        else:
            st.success("No tumour detected")

st.caption("Educational project only. Not for medical diagnosis.")
