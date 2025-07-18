import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
from PIL import Image
import matplotlib.cm as cm
import io
import base64
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'penumonia_model.h5') 
model_path = os.path.abspath(model_path) 
model = load_model(model_path, compile = False)

# Grad-CAM Function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Apply heatmap on image
def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap_color * alpha + img
    return np.uint8(superimposed)

# Streamlit UI
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will classify it as **Normal** or **Pneumonia** with Grad-CAM.")

uploaded_file = st.file_uploader("Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Pneumonia" if prediction > 0.5 else "Normal"
    confidence = float(prediction)

    st.subheader("Prediction:")
    if label == "Pneumonia":
        st.error(f"⚠️ Pneumonia Detected — Confidence: {confidence:.4f}")
    else:
        st.success(f"✅ Normal — Confidence: {1 - confidence:.4f}")

    # Grad-CAM
    last_conv_layer = "conv5_block16_concat"  # <- replace based on model inspection
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

    img_for_overlay = np.array(image.resize((224, 224)))
    superimposed_img = overlay_heatmap(img_for_overlay, heatmap)

    st.image(superimposed_img, caption="Grad-CAM", use_column_width=True)

    # Save prediction report as text
    report_text = f"""Prediction Report:
-------------------
Label: {label}
Confidence: {confidence:.4f}
Model: {model.name}
"""

    # Download report button
    buffer = io.BytesIO()
    buffer.write(report_text.encode())
    buffer.seek(0)
    st.download_button("📄 Download Report", buffer, file_name="prediction_report.txt", mime="text/plain")
