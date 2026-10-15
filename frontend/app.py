import os
import sys
import base64
import io
import streamlit as st
from PIL import Image
import requests

# Settings (override with environment variables)
# Base URL of the FastAPI backend.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
# Seconds to wait for the backend. The first request also loads the model, so keep it generous.
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))
# Optional background image for the page. Defaults to image.jpg in the repository root.
BACKGROUND_IMAGE = os.environ.get(
    "BACKGROUND_IMAGE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "image.jpg"),
)


def get_error_message(response):
    """Return a readable error message, whether or not the backend replied with JSON."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if not detail:
        detail = response.text.strip()[:500] or response.reason or "Unknown error"
    return f"{response.status_code}: {detail}"


# Streamlit UI
def streamlit_ui():
    # Get base64 of background image
    def get_base64_of_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    # Try to get background image, use fallback if not found
    try:
        bg_image = get_base64_of_image(BACKGROUND_IMAGE)
    except OSError:
        # Fallback to no background if image not found
        bg_image = None
        
    # CSS styling with conditional background
    bg_style = f"""
        .stApp {{
            background: url("data:image/jpg;base64,{bg_image}") no-repeat center center fixed;
            background-size: cover;
        }}
    """ if bg_image else ""
    
    st.markdown(f"""
        <style>
            {bg_style}

            .block-container {{
                padding-left: 30px !important;
                padding-right: 30px !important;
                max-width: 100% !important;
            }}

            .title {{
                font-size: 50px;
                font-weight: 550;
                color: #4CAF50;
                margin-bottom: 5px;
            }}

            .image-box {{
                border: 2px solid #ddd;
                padding: 15px;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.8);
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                text-align: center;
                width: 100%;
            }}

            .stAlert {{
                color: #66BB6A !important;
                font-weight: bold !important;
            }}
            
            .sidebar .image-selector {{
                padding: 10px;
                border-radius: 5px;
                background-color: rgba(255, 255, 255, 0.8);
                margin-bottom: 15px;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state for storing images
    if 'input_image' not in st.session_state:
        st.session_state.input_image = None
    if 'output_image' not in st.session_state:
        st.session_state.output_image = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Title
    st.markdown('<p class="title">Image Colorization with Stable Diffusion</p>', unsafe_allow_html=True)
    
    # Create a sidebar for parameters
    with st.sidebar:
        st.markdown("## Colorization Parameters")
        prompt = st.text_input("Prompt", "Colorize this image")
        a_prompt = st.text_input("Added Prompt", "best quality, natural colors")
        
        with st.expander("Advanced Options", expanded=False):
            n_prompt = st.text_input("Negative Prompt", 
                                     "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")
            image_resolution = st.slider("Image Resolution", 256, 768, 512, 64)
            ddim_steps = st.slider("Steps", 1, 100, 20)
            strength = st.slider("Control Strength", 0.0, 2.0, 1.0, 0.01)
            scale = st.slider("Guidance Scale", 0.1, 30.0, 9.0, 0.1)
            seed = st.slider("Seed", -1, 2147483647, -1)
            eta = st.number_input("eta (DDIM)", 0.0)
            guess_mode = st.checkbox("Guess Mode", False)
    
    # Main content area
    st.markdown("### Upload a grayscale image")
    uploaded_file = st.file_uploader("Choose a grayscale image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Store input image
        image = Image.open(uploaded_file)
        st.session_state.input_image = image
        
        # Process button
        if st.button("Colorize Image"):
            st.session_state.processing = True
            
            # Create params for processing
            params = {
                "prompt": prompt,
                "a_prompt": a_prompt,
                "n_prompt": n_prompt,
                "num_samples": 1,
                "image_resolution": image_resolution,
                "ddim_steps": ddim_steps,
                "guess_mode": guess_mode,
                "strength": strength,
                "scale": scale,
                "seed": seed,
                "eta": eta
            }
            
            with st.spinner("🔄 Colorizing..."):
                # Send request to backend API
                files = {"file": uploaded_file.getvalue()}
                try:
                    response = requests.post(f"{BACKEND_URL}/api/colorize", files=files, data=params,
                                             timeout=REQUEST_TIMEOUT)
                except requests.exceptions.Timeout:
                    response = None
                    st.error(f"Error: the backend did not answer within {REQUEST_TIMEOUT:.0f} seconds.")
                except requests.exceptions.RequestException as e:
                    response = None
                    st.error(f"Error: could not reach the backend at {BACKEND_URL} ({e}).")

                if response is None:
                    st.session_state.processing = False
                elif response.status_code == 200:
                    st.session_state.output_image = Image.open(io.BytesIO(response.content))
                    st.session_state.processing = False
                else:
                    st.error(f"Error: {get_error_message(response)}")
                    st.session_state.processing = False
        
        # Display both images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Original Image (B&W)")
            if st.session_state.input_image is not None:
                st.image(st.session_state.input_image, use_column_width=True)
        
        with col2:
            st.markdown("### 🎨 Colorized Image")
            if st.session_state.output_image is not None:
                st.image(st.session_state.output_image, use_column_width=True)
                
                # Add download button for colorized image
                buf = io.BytesIO()
                st.session_state.output_image.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    label="Download Colorized Image",
                    data=buf,
                    file_name="colorized_image.png",
                    mime="image/png"
                )
            elif st.session_state.processing:
                st.info("Processing... please wait")
            else:
                st.info("Click 'Colorize Image' to generate a colorized version")

if __name__ == "__main__":
    streamlit_ui()