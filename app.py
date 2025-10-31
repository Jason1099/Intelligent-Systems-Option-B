from io import BytesIO
from PIL import Image
import os
import tempfile
import numpy as np
import streamlit as st

import os
import tempfile
from io import BytesIO
from PIL import Image
import numpy as np
import streamlit as st
# --- Import the core logic from main.py ---
from Models.digit_recognition_system import run 
# ... (rest of imports and setup) ...

def run_digit_recognition(image_bytes: bytes, model_kind: str):
    """
    Saves the image bytes to a temporary file and runs the recognition system.
    Returns a formatted string containing all line results (expressions/digits).
    """
    temp_path = None
    try:
        # 1. Save the uploaded image bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name
        
        # 2. Call the core recognition logic
        results = run(
            image_path=temp_path,
            kind=model_kind,
            out_dir="digits_export"
        )
        print("Prediction Complete!")

        # 3. Clean up the temporary file
        os.remove(temp_path)
        temp_path = None

        # 4. Process results for display
        line_results = results.get("line_results", [])
        digit_results = results.get("digit_results", [])

        # --- REVISED RESULT PROCESSING LOGIC ---
        if "ext" in model_kind:
            # Handle Extension (Math) model: Return ALL expressions and results
            if line_results:
                output_lines = ["**Recognised Math Expressions:**"]
                # Sort by line index for clean display
                for lr in sorted(line_results, key=lambda x: x.get("line_index", 0)):
                    li = lr.get("line_index", 0)
                    expr = lr.get("expression", "?")
                    val = lr.get("result", None)

                    if val is not None:
                         output_lines.append(f"Line {li}: `{expr}` = **{val}**")
                    else:
                         output_lines.append(f"Line {li}: `{expr}` (No value)")

                return "\n\n".join(output_lines)
            else:
                return "Math: No expressions recognised."
        
        else:
            # Handle Main (Digit) model: Return structured expression results
            if line_results:
                output_lines = ["**Recognised Digits/Expression:**"]
                # Sort by line index
                for lr in sorted(line_results, key=lambda x: x.get("line_index", 0)):
                    li = lr.get("line_index", 0)
                    expr = lr.get("expression", "?")
                    val = lr.get("result", None) # Result might be None if it's just a sequence of digits

                    if val is not None and str(val) != expr:
                         # Display as an expression with result if a calculation was performed
                         output_lines.append(f"Line {li}: `{expr}` = **{val}**")
                    elif expr != '?':
                         # Display as a sequence of digits/symbols
                         output_lines.append(f"Line {li}: **{expr}**")
                    else:
                         output_lines.append(f"Line {li}: No clear sequence found.")

                return "\n\n".join(output_lines)
            elif digit_results:
                 # Fallback for single-component images without line grouping
                 first_digit = digit_results[0].get('digit', '?')
                 return f"**Predicted Digit:** {first_digit}"
            
            return "HNRS: No result found"


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        # Use a more generic error message for the pill
        return "ERROR"
    finally:
        # Ensure cleanup if an error occurred before os.remove was called
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

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

        if main_predict_clicked:
            if not st.session_state.main_file_bytes:
                st.warning("Please upload an image first.")
            else:
                model_name = st.session_state.main_model_choice
                
                # Map Streamlit choice to the model_kind string used in main.py
                model_kind = "cnn" if model_name == "CNN" else "vit"
                
                # Run the prediction using the new wrapper function
                with st.spinner(f"Running {model_name} recognition..."):
                    pred = run_digit_recognition(st.session_state.main_file_bytes, model_kind)

                st.session_state.main_pred = pred

        # Check if it's the default placeholder, otherwise display the structured result
        if st.session_state.main_pred == "-":
             st.markdown(f'<div class="pill">{st.session_state.main_pred}</div>', unsafe_allow_html=True)
        else:
             # Use st.markdown to render the multi-line result with formatting
             st.markdown(st.session_state.main_pred)

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
                # The extension model in your main.py is "vit_ext"
                model_kind = "vit_ext"
                
                # Run the prediction using the new wrapper function
                with st.spinner("Running ViT Extension recognition..."):
                    pred = run_digit_recognition(st.session_state.ext_file_bytes, model_kind)

                st.session_state.ext_pred = pred
        # Check if it's the default placeholder, otherwise display the multi-line result
        if st.session_state.ext_pred == "-":
             st.markdown(f'<div class="pill">{st.session_state.ext_pred}</div>', unsafe_allow_html=True)
        else:
             # Use st.markdown to render the multi-line result with formatting
             st.markdown(st.session_state.ext_pred)

# ------------------- ABOUT TAB -------------------
with tabs[2]:
    st.markdown("## About")
    st.write("Work in progress. Coming soon.")