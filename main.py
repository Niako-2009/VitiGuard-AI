import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="VitiGuard AI", page_icon="🍇")
st.title("🍇 VitiGuard AI — Vineyard Disease Detection")
st.write("Upload a grape leaf image and the AI will diagnose it instantly.")

MODEL_PATH = "model/vitiguard_model.h5"

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("Brain file not found! Make sure 'vitiguard_model.h5' is inside a folder named 'model'.")
        return None

model = load_my_model()

uploaded_file = st.file_uploader("Upload a grape leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)
        classes = ['Black Rot', 'Esca', 'Healthy', 'Leaf Blight']
        result = classes[np.argmax(prediction)]
    
    st.success(f"### Diagnosis: {result}")
    
    if result != "Healthy":
        st.info("💡 **Eco-Friendly Tip:** Increase airflow between vines and remove infected leaves immediately.")
