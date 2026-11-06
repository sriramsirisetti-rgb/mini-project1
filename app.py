import json
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from advice import get_advice

IMG = 224
CONF_THRESHOLD = 0.60

st.set_page_config(page_title="Leaf Doctor", page_icon="🌿", layout="centered")


@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("model/plant_model.keras", compile=False)
    classes = json.load(open("model/class_names.json"))
    return model, classes


def pretty(label: str):
    crop, _, disease = label.partition("___")
    return crop.replace("_", " "), disease.replace("_", " ").strip() or "healthy"


def predict(model, classes, img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG, IMG))
    x = np.expand_dims(np.array(img, dtype="float32"), 0)   # raw 0-255; model preprocesses internally
    probs = model.predict(x, verbose=0)[0]
    top = probs.argsort()[::-1][:3]
    return [(classes[i], float(probs[i])) for i in top]


st.title("🌿 Leaf Doctor")
st.caption("Take or upload a clear photo of a single crop leaf to check for disease.")

try:
    model, classes = load_model()
except Exception as e:
    st.error("Model not found. Run `python train.py` first to create `model/plant_model.keras`.")
    st.stop()

tab_cam, tab_up = st.tabs(["📷 Camera", "🖼️ Upload"])
with tab_cam:
    cam = st.camera_input("Take a photo of the leaf")
with tab_up:
    up = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

file = cam or up
if file:
    image = Image.open(file)
    st.image(image, caption="Your photo", use_container_width=True)

    with st.spinner("Analysing leaf..."):
        results = predict(model, classes, image)

    label, conf = results[0]
    crop, disease = pretty(label)

    if conf < CONF_THRESHOLD:
        st.warning("I'm not confident about this photo. Try again with a single leaf, "
                   "good daylight, and a plain background.")
    else:
        if disease == "healthy":
            st.success(f"**{crop}: healthy** ({conf:.0%} confidence)")
        else:
            st.error(f"**{crop}: {disease}** ({conf:.0%} confidence)")
        st.subheader("What you can do")
        st.write(get_advice(disease))

    with st.expander("Other possibilities"):
        for lbl, p in results:
            c, d = pretty(lbl)
            st.write(f"{c} - {d}: {p:.1%}")
            st.progress(min(p, 1.0))

st.divider()
st.caption("This tool gives an AI estimate. Confirm with a local agriculture expert before treating.")
