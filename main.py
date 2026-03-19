import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from datasets import load_dataset
from fpdf import FPDF

st.set_page_config(page_title="VitiGuard AI 🍇", layout="wide")

os.makedirs("model", exist_ok=True)
MODEL_PATH = "model/vitiguard_model.h5"

CLASSES = ["Black_Rot", "ESCA", "Leaf_Blight", "Healthy"]

DISEASE_INFO = {
    "Black_Rot": {
        "symptoms": "Brown circular spots on leaves and shriveled dark berries.",
        "solution": "Use copper-based organic fungicides and remove infected vines.",
        "pesticide_reduction": 40
    },
    "ESCA": {
        "symptoms": "Tiger stripe patterns on leaves and dark fruit spots.",
        "solution": "Protect pruning wounds and avoid pruning during wet weather.",
        "pesticide_reduction": 65
    },
    "Leaf_Blight": {
        "symptoms": "Large irregular brown patches and early leaf drop.",
        "solution": "Improve airflow and apply organic compost treatments.",
        "pesticide_reduction": 30
    },
    "Healthy": {
        "symptoms": "Bright green leaves without spots or damage.",
        "solution": "No treatment required. Maintain irrigation and soil health.",
        "pesticide_reduction": 100
    }
}


def preprocess(image):
    image = image.resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def generate_heatmap(model, img_array):
    base_model = model.layers[0]
    target_layer = base_model.get_layer("top_activation")

    grad_model = tf.keras.models.Model(
        [base_model.inputs],
        [target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


def train_model():
    st.info("Loading dataset from HuggingFace...")
    ds = load_dataset("adamkatchee/grape-leaf-disease-augmented-dataset", split="train", streaming=True)

    images = []
    labels = []

    for item in ds.take(300):
        img = item["image"].convert("RGB").resize((224, 224))
        label = item["label"]
        if label < 4:
            images.append(np.array(img) / 255.0)
            labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(4, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X, y, epochs=5, batch_size=16, verbose=0)
    model.save(MODEL_PATH)
    return model


def generate_pdf(disease, confidence, info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "VitiGuard AI Diagnosis Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Disease: {disease}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Symptoms: {info['symptoms']}")
    pdf.multi_cell(0, 10, f"Treatment: {info['solution']}")
    pdf.multi_cell(0, 10, f"Pesticide Reduction: {info['pesticide_reduction']}%")
    return pdf.output(dest="S").encode("latin-1")


st.title("🍇 VitiGuard AI — Vineyard Disease Detection")
st.write("Upload a grape leaf image and the AI will diagnose disease and recommend eco-friendly treatment.")

model = load_model()

if model is None:
    if st.button("Train AI Model"):
        with st.spinner("Training AI..."):
            model = train_model()
            st.success("Model trained successfully!")
            st.rerun()
else:
    uploaded = st.file_uploader("Upload grape leaf image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess(image)
        preds = model.predict(img_array)
        idx = np.argmax(preds)
        confidence = preds[0][idx] * 100
        disease = CLASSES[idx]
        info = DISEASE_INFO[disease]

        with col2:
            st.success(f"Detected: {disease}")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.write("### Symptoms")
            st.write(info["symptoms"])
            st.write("### Eco Friendly Treatment")
            st.write(info["solution"])
            st.write("### Pesticide Reduction")
            st.progress(info["pesticide_reduction"] / 100)

        heatmap = generate_heatmap(model, img_array)
        st.subheader("AI Attention Heatmap")

        fig, ax = plt.subplots()
        ax.imshow(image)
        resized_heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image.size[1], image.size[0]))
        ax.imshow(resized_heatmap, cmap="jet", alpha=0.4)
        ax.axis("off")
        st.pyplot(fig)

        pdf_bytes = generate_pdf(disease, confidence, info)
        st.download_button(
            "Download AI Diagnosis Report",
            pdf_bytes,
            file_name=f"vitiguard_{disease}_report.pdf",
            mime="application/pdf"
        )
