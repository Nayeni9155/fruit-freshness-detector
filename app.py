import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ✅ Must be the FIRST Streamlit command
st.set_page_config(page_title="Fruit Freshness Detector", layout="centered")

# =====================
# Load Model (Cached)
# =====================
@st.cache_resource
def load_model(model_path):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Path to trained model
MODEL_PATH = r"C:\Users\Hello\dhaval\intership\2\best_resnet50_fresh_spoiled.pth"
model = load_model(MODEL_PATH)

# =====================
# Image Preprocessing
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# UI
# =====================
st.title("🍋 Fruit Freshness Classifier")
st.write("Upload a fruit image (e.g., lemon) to predict whether it's **Fresh** or **Spoiled**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output > 0.5).float().item()

    label = "🟢 Fresh Fruit" if prediction == 0 else "🔴 Spoiled Fruit"
    st.markdown(f"## Prediction: {label}")
