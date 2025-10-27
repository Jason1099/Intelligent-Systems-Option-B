from io import BytesIO
from PIL import Image
import numpy as np
import streamlit as st

st.set_page_config(page_title="HNRS + Math (ViT)", page_icon=" ", layout="wide")

# CSS styling for UI elements
st.markdown("""
<style>
.pill {
  display:inline-block; padding:10px 14px; border-radius:12px;
  background:#dcdcdc; font-weight:800;
}
.hint { color:#334155; opacity:.9; }
.card {
  border:1px solid #cbd5e1; border-radius:8px; padding:10px;
  text-align:center; background:#d3d3d3; font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# Initialize Streamlit session state variables
def init_state():
    # main tab state
    st.session_state.setdefault("main_file_bytes", None)
    st.session_state.setdefault("main_pred", "–")
    st.session_state.setdefault("main_model_choice", "CNN")  # default model
    st.session_state.setdefault("main_models_loaded", {"CNN": False, "ViT": False})
    st.session_state.setdefault("main_models", {"CNN": None, "ViT": None})

    # extension tab state
    st.session_state.setdefault("ext_file_bytes", None)
    st.session_state.setdefault("ext_pred", "–")

# Functions to clear stored file/prediction states
def clear_main_state():
    st.session_state.main_file_bytes = None
    st.session_state.main_pred = "–"

def clear_ext_state():
    st.session_state.ext_file_bytes = None
    st.session_state.ext_pred = "–"

init_state()

st.markdown("### Intelligent Systems – Project B (Group 5)")
tabs = st.tabs(["Main (HNRS)", "Extension (Math)", "About"])

# ------------------- MAIN TAB (HNRS) -------------------
with tabs[0]:
    st.markdown("## Handwritten Number Recognition System")

    col_left, col_mid, col_right = st.columns([1, 2, 1], gap="large")

    # Left column - buttons and model info
    with col_left:
        st.write("#### Controls")

        # Model selector
        st.caption("Select Model")
        st.session_state.main_model_choice = st.radio(
            "Model",
            options=["CNN", "ViT"],
            index=["CNN", "ViT"].index(st.session_state.main_model_choice),
            label_visibility="collapsed",
            key="main_model_selector",
        )

        # Clear button
        if st.button("Clear", key="main_clear_btn", type="secondary", use_container_width=True):
            clear_main_state()
            st.rerun()

        st.write("")
        st.caption("Model Info")
        # Dynamic card reflecting selected model
        selected_model_label = "Convolutional Neural Network (CNN)" if st.session_state.main_model_choice == "CNN" else "Vision Transformer (ViT)"
        st.markdown(f'<div class="card">Model: {selected_model_label}</div>', unsafe_allow_html=True)

        # Predict button
        main_predict_clicked = st.button("Recognise", key="main_predict_btn", use_container_width=True)

    # Middle column - image uploader and preview
    with col_mid:
        st.write("#### Upload image")
        st.caption("Drag & drop digit images (0–9) or click to upload.")

        up_main = st.file_uploader("  ", type=["png", "jpg", "jpeg"], key="main_uploader", label_visibility="collapsed")
        if up_main is not None:
            st.session_state.main_file_bytes = up_main.getvalue()

        if st.session_state.main_file_bytes:
            img_main = Image.open(BytesIO(st.session_state.main_file_bytes))
            st.image(img_main, use_container_width=True)
        else:
            st.markdown('<div class="hint">Drag & Drop pics in or Upload</div>', unsafe_allow_html=True)

    # Right column - prediction result
    with col_right:
        st.write("#### Result")

        # Lazy-load selected model (replace with your actual loaders)
        def load_cnn():
            # TODO: load and return your CNN model object
            return "cnn_model_obj"

        def load_cvit():
            # TODO: load and return your cViT model object
            return "cvit_model_obj"
        # Simple preprocessor (replace with your actual preprocessing)
        def preprocess_pil_for_model(img: Image.Image, model_name: str):
            # Example: convert to grayscale and resize to 28x28 for CNN; 224x224 for cViT
            if model_name == "CNN":
                return img.convert("L").resize((28, 28))
            else:
                return img.convert("RGB").resize((224, 224))

        # Simple predictor stubs (replace with your actual inference)
        def predict_with_cnn(model, img_proc):
            # TODO: run your CNN forward pass; return predicted digit as string
            return "CNN: (pending backend)"

        def predict_with_cvit(model, img_proc):
            # TODO: run your cViT forward pass; return predicted digit as string
            return "cViT: (pending backend)"

        if main_predict_clicked:
            if not st.session_state.main_file_bytes:
                st.warning("Please upload an image first.")
            else:
                model_name = st.session_state.main_model_choice

                # Load once per model name
                if not st.session_state.main_models_loaded[model_name]:
                    if model_name == "CNN":
                        st.session_state.main_models["CNN"] = load_cnn()
                    else:
                        st.session_state.main_models["cViT"] = load_cvit()
                    st.session_state.main_models_loaded[model_name] = True

                # Preprocess and predict
                img_obj = Image.open(BytesIO(st.session_state.main_file_bytes))
                proc = preprocess_pil_for_model(img_obj, model_name)

                if model_name == "CNN":
                    pred = predict_with_cnn(st.session_state.main_models["CNN"], proc)
                else:
                    pred = predict_with_cvit(st.session_state.main_models["cViT"], proc)

                st.session_state.main_pred = pred

        st.markdown(f'<div class="pill">{st.session_state.main_pred}</div>', unsafe_allow_html=True)

# ------------------- EXTENSION TAB (Math) -------------------
with tabs[1]:
    st.markdown("## Extension (Handwritten Math)")

    col_left, col_mid, col_right = st.columns([1, 2, 1], gap="large")

    # Left column - buttons and model info
    with col_left:
        st.write("#### Controls")

        if st.button("Clear", key="ext_clear_btn", type="secondary", use_container_width=True):
            clear_ext_state()
            st.rerun()

        st.write("")
        st.caption("Model")
        st.markdown('<div class="card">Model: Vision Transformer (ViT)</div>', unsafe_allow_html=True)

        ext_predict_clicked = st.button("Recognise", key="ext_predict_btn", use_container_width=True)

    # Middle column - image uploader and preview
    with col_mid:
        st.write("#### Upload image")
        st.caption("Drag & drop equation images or click to upload.")

        up_ext = st.file_uploader("  ", type=["png", "jpg", "jpeg"], key="ext_uploader", label_visibility="collapsed")
        if up_ext is not None:
            st.session_state.ext_file_bytes = up_ext.getvalue()

        if st.session_state.ext_file_bytes:
            img_ext = Image.open(BytesIO(st.session_state.ext_file_bytes))
            st.image(img_ext, use_container_width=True)
        else:
            st.markdown('<div class="hint">Drag & Drop pics in or Upload</div>', unsafe_allow_html=True)

    # Right column - prediction result
    with col_right:
        st.write("#### Result")

        if ext_predict_clicked:
            if not st.session_state.ext_file_bytes:
                st.warning("Please upload an image first.")
            else:
                # TODO: replace this with actual prediction from your math model
                st.session_state.ext_pred = "ViT: (pending backend)"

        st.markdown(f'<div class="pill">{st.session_state.ext_pred}</div>', unsafe_allow_html=True)

# ------------------- ABOUT TAB -------------------
with tabs[2]:
    st.markdown("## About")
    st.write("Work in progress. Coming soon.")
