import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow import keras
import os
import tempfile

# Page configuration
st.set_page_config(
    page_title="Image Tampering Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .title-text {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        font-size: 18px;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f4ff;
        border-left: 5px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .result-real {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
    }
    .result-fake {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="title-text">🔍 Image Tampering Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Detect if an image has been tampered with using ELA (Error Level Analysis) & CNN</div>', unsafe_allow_html=True)

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

# Sidebar information
with st.sidebar:
    st.markdown("### 📋 About This Tool")
    st.markdown("""
    **How it works:**
    - **ELA (Error Level Analysis)**: Highlights areas with different compression levels
    - **CNN Model**: Analyzes the ELA image to detect tampering patterns
    - **Dataset**: Trained on CASIA2 dataset
    
    **Interpretation:**
    - **Real Image**: Consistent error levels across the image
    - **Tampered Image**: High inconsistency in error levels in edited areas
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.info("""
    - **Model Type**: Convolutional Neural Network (CNN)
    - **Input Size**: 128 × 128 pixels
    - **Training Data**: CASIA2
    - **Output**: Real/Fake classification with confidence
    """)

# Main content area
st.markdown("---")

# Upload section with better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )

with col2:
    st.markdown("### 📊 Processing Steps")
    st.markdown("""
    1. Original image upload
    2. ELA image generation
    3. Model prediction
    4. Result display
    """)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Processing indicator
        with st.spinner("🔄 Processing image..."):
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
        
        st.markdown("---")
        
        # Display images side by side with better styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Original Image")
            st.image(original_image, use_column_width=True)
        
        with col2:
            st.markdown("#### 🔍 ELA Image")
            st.image(ela_image, use_column_width=True)
        
        # Display results with better styling
        st.markdown("---")
        st.markdown("### 📈 Analysis Results")
        
        # Create result display
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if result == 'Real':
                st.markdown(
                    '<div class="result-real"><h2>✅ REAL IMAGE</h2><p style="font-size: 24px; margin: 0;">Authentic</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-fake"><h2>⚠️ FAKE IMAGE</h2><p style="font-size: 24px; margin: 0;">Tampered/Manipulated</p></div>',
                    unsafe_allow_html=True
                )
        
        with res_col2:
            st.markdown(f"""
            <div style="background-color: #e8f4f8; padding: 20px; border-radius: 5px; border-left: 5px solid #0088cc;">
                <h3 style="margin-top: 0; color: #0088cc;">Confidence Score</h3>
                <p style="font-size: 32px; font-weight: bold; color: #0088cc; margin: 0;">{confidence}%</p>
                <p style="color: #666; margin-top: 10px;">Prediction certainty level</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional analysis info
        st.markdown("---")
        st.markdown("### 💡 What This Means")
        
        if result == 'Real':
            st.success("""
            The image appears to be **authentic**. The error levels are consistent across the image,
            indicating no significant tampering was detected by the model.
            """)
        else:
            st.warning("""
            The image appears to be **tampered**. Inconsistent error levels were detected,
            suggesting that parts of the image may have been edited or manipulated.
            """)
        
        st.info("""
        **Note:** This analysis is based on Error Level Analysis (ELA) and CNN predictions.
        While highly accurate, no detection method is 100% foolproof. The confidence score
        indicates the model's certainty in its prediction.
        """)
        
        # Clean up temp files
        os.remove(tmp_path)
        os.remove(ela_path)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.info("Please try uploading a different image file.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>🛡️ Image Tampering Detection System</strong></p>
    <p>Powered by Error Level Analysis (ELA) & Convolutional Neural Networks (CNN)</p>
    <p>Trained on CASIA2 dataset | Model: Keras/TensorFlow</p>
    <p style="font-size: 12px; margin-top: 20px;">© 2024 - For educational and research purposes</p>
</div>
""", unsafe_allow_html=True)
