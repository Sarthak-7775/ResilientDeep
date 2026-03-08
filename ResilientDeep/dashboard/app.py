# dashboard/app.py
import streamlit as st
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add the root directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.modules.model import ResilientDetector
from src.data_pipeline.dataset import baseline_transforms

st.title("ResilientDeep Prototype")
st.write("Upload an image to test against the Visibility Matrix and High-Frequency Enhancer.")

@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResilientDetector(num_classes=2).to(device)
    
    # Path to your best saved weights (assuming execution via main.py)
    weight_path = os.path.abspath("models/checkpoints/best_model.pth")
    
    if os.path.exists(weight_path):
        # Load the dictionary into the skeleton
        model.load_state_dict(torch.load(weight_path, map_location=device))
        st.sidebar.success("Successfully loaded trained weights from best_model.pth.")
    else:
        st.sidebar.warning("No trained weights found. Using randomized initialization. Please run the training pipeline first.")
        
    model.eval() # Set to evaluation mode (crucial for inference)
    return model, device

# Load the model state
model, device = load_trained_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analyzing for invisible compression artifacts...")
    
    # Preprocess
    input_tensor = baseline_transforms(image).unsqueeze(0).to(device)
    
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