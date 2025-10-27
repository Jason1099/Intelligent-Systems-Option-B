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
    st.session_state.setdefault("main_file_bytes", None)
    st.session_state.setdefault("main_pred", "–")
    st.session_state.setdefault("ext_file_bytes", None)
    st.session_state.setdefault("ext_pred", "–")
    st.session_state.setdefault("model", None)
    st.session_state.setdefault("model_loaded", False)

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

        if st.button("Clear", key="main_clear_btn", type="secondary", use_container_width=True):
            clear_main_state()
            st.rerun()

        st.write("")
        st.caption("Model")
        st.markdown('<div class="card">Model: Vision Transformer (ViT)</div>', unsafe_allow_html=True)

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

        if main_predict_clicked:
            if not st.session_state.main_file_bytes:
                st.warning("Please upload an image first.")
            else:
                # TODO: replace this with actual prediction from your model
                st.session_state.main_pred = "ViT: (pending backend)"

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
                # TODO: replace this with actual prediction from your model
                st.session_state.ext_pred = "ViT: (pending backend)"

        st.markdown(f'<div class="pill">{st.session_state.ext_pred}</div>', unsafe_allow_html=True)

# ------------------- ABOUT TAB -------------------
with tabs[2]:
    st.markdown("## About")
    st.write("Work in progress. Coming soon.")
