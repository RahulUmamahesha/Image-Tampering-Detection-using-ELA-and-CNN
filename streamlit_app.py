import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow import keras
import os
import tempfile

# Page configuration
st.set_page_config(page_title="Image Tampering Detection", layout="wide")

st.title("🔍 Image Tampering Detection")
st.markdown("Detect if an image has been tampered with using ELA (Error Level Analysis) & CNN")

# Load model
@st.cache_resource
def load_model():
    return keras.models.load_model('model_casia_run1.h5')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def convert_to_ela_image(image_path, quality=90):
    """Convert image to ELA image"""
    temp_filename = 'temp_file_name.jpg'
    
    # Open the image and save a temporary JPEG version with specified quality
    image = Image.open(image_path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    # Compute the difference between the original and the temporary image
    ela_image = ImageChops.difference(image, temp_image)
    
    # Get the extrema of the ELA image
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    # Normalize the ELA image to the maximum difference
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Clean up temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    return ela_image

def prepare_image(image_path):
    """Prepare image for prediction"""
    image_size = (128, 128)
    ela_image = Image.open(image_path).resize(image_size)
    
    # Convert ELA image to a NumPy array and normalize
    ela_array = np.array(ela_image).flatten() / 255.0
    
    return ela_array

def predict_fake_real(ela_image_path):
    """Make prediction on image"""
    image_array = prepare_image(ela_image_path)
    image_array = image_array.reshape(1, 128, 128, 3)
    prediction = model.predict(image_array, verbose=0)
    
    confidence = round(np.max(prediction) * 100, 2)
    result = 'Fake' if np.argmax(prediction) == 0 else 'Real'
    
    return confidence, result

# UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

with col2:
    st.subheader("Information")
    st.info("""
    **How it works:**
    - Error Level Analysis (ELA) highlights areas with different compression levels
    - CNN model analyzes the ELA image to detect tampering
    - Fake images show high inconsistency in error levels
    """)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Original image
        original_image = Image.open(tmp_path)
        
        # Convert to ELA
        ela_image = convert_to_ela_image(tmp_path)
        
        # Save ELA temporarily for prediction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_ela:
            ela_image.save(tmp_ela.name)
            ela_path = tmp_ela.name
        
        # Make prediction
        confidence, result = predict_fake_real(ela_path)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
        
        with col2:
            st.subheader("ELA Image")
            st.image(ela_image, use_column_width=True)
        
        # Results
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if result == 'Real':
                st.success(f"### ✅ Result: {result}")
            else:
                st.error(f"### ❌ Result: {result}")
        
        with col2:
            st.info(f"### Confidence: {confidence}%")
        
        # Clean up temp files
        os.remove(tmp_path)
        os.remove(ela_path)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        os.remove(tmp_path)

st.markdown("---")
st.markdown("**About**: This model uses CNN trained on CASIA2 dataset for image tampering detection")
