# dashboard/app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Set the page layout and app metadata
st.set_page_config(page_title="ResilientDeep Prototype", layout="wide")

# Add the root directory to the path so we can import our modules
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from src.modules.model import ResilientDetector
    from src.data_pipeline.dataset import baseline_transforms
except Exception as e:
    st.error(f"Failed to import dashboard dependencies: {e}")
    st.stop()

st.title("ResilientDeep Prototype")
st.write("Upload an image to test against the Visibility Matrix and High-Frequency Enhancer.")

@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResilientDetector(num_classes=2).to(device)
    
    # Path to your best saved weights (assuming execution via main.py)
    weight_path = ROOT_DIR / "models" / "checkpoints" / "model_epoch_3.pth"
    
    if weight_path.exists():
        # Load the dictionary into the skeleton
        model.load_state_dict(torch.load(weight_path, map_location=device))
        st.sidebar.success(f"Successfully loaded trained weights from {weight_path.name}.")
    else:
        st.sidebar.error("No trained weights found. Please run the training pipeline first and reload the app.")
        st.stop()
        
    model.eval() # Set to evaluation mode (crucial for inference)
    return model, device

# Load the model state
with st.spinner("Loading model..."):
    model, device = load_trained_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analyzing for invisible compression artifacts...")
    
    # --- THE FIX IS HERE ---
    # Convert the PIL image to a NumPy array so ToPILImage() doesn't crash
    image_np = np.array(image)
    
    # Preprocess using the numpy array
    input_tensor = baseline_transforms(image_np).unsqueeze(0).to(device)
    # -----------------------
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        fake_prob = probabilities[0][1].item() * 100
        
    st.subheader(f"Fake Probability: {fake_prob:.2f}%")
    
    if fake_prob > 50:
        st.error("Verdict: MANIPULATED (ShallowReal detected)")
    else:
        st.success("Verdict: AUTHENTIC")