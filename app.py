import streamlit as st
import numpy as np
import cv2
import os

from utils.model_loader import load_model, predict_dementia, get_prediction_probabilities
from utils.preprocessing import preprocess_uploaded_image
from utils.report_generator import generate_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Dementia Detection using MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD STYLES
# --------------------------------------------------
def load_custom_css():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_custom_css()

# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------
# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

# User credentials
USERS = {
    "admin": "admin123",
    "doctor": "doctor123",
    "user": "user123"
}

def render_navbar():
    user_status = ""
    if st.session_state.authenticated:
        user_status = f'<div class="navbar-user"><span class="user-badge"><i class="fa-solid fa-user-md"></i> {st.session_state.username.upper()}</span></div>'
    navbar_html = f'<div class="navbar"><div class="navbar-brand"><i class="fa-solid fa-brain"></i> AI Dementia Detection <span>System</span></div>{user_status}</div>'
    st.markdown(navbar_html, unsafe_allow_html=True)

def login_page():
    """Display modern login page"""
    render_navbar()
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown('<div class="hero-container"><div class="hero-tag">AI-Powered Neuro-Diagnostics</div><h1>Precision <span>Dementia</span> Detection Platform</h1><p>Leveraging deep residual network architectures (ResNet50) to evaluate brain MRI scans and identify cognitive decline stages. Get automated patient reports, class probability distribution maps, and direct diagnostic recommendations.</p></div>', unsafe_allow_html=True)
        
        # Display illustration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        hero_img_path = os.path.join(base_dir, "assets", "hero_illustration.png")
        if os.path.exists(hero_img_path):
            st.image(hero_img_path, use_container_width=True)
            
    with col2:
        st.markdown('<div class="login-header"><h2>Authorized Portal</h2><p>Enter your credentials to access diagnostic tools</p></div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login to Workspace", use_container_width=True)
            
            if submit_button:
                if username in USERS and USERS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"✅ Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

def logout():
    """Logout function"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
def main_app():
    """Main application after login"""
    # Sidebar with user info and logout
    with st.sidebar:
        st.markdown(f"### 👤 Logged in as: **{st.session_state.username}**")
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        st.markdown("---")
        st.markdown("### 📋 Quick Info")
        st.info("Upload an MRI image to analyze dementia stage using AI.")
    
    render_navbar()
    
    st.title("🧠 Dementia Detection using MRI Images")
    st.markdown("Upload an MRI image to predict dementia stage with explainable AI (Grad-CAM).")
    
    # Patient Information Section
    with st.expander("👤 Patient Information (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
            age = st.number_input("Age", min_value=0, max_value=120, value=0)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Not Specified"])
            clinical_notes = st.text_area("Clinical Notes/Symptoms", placeholder="Enter clinical notes or symptoms (optional)")
    
    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    model = load_model("resnet50_dementia_model.h5")
    
    # --------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------
    uploaded_file = st.file_uploader(
        "📤 Upload MRI Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI image in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file:
        # --------------------------------------------------
        # READ ORIGINAL IMAGE (FOR DISPLAY & GRADCAM)
        # --------------------------------------------------
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_img is None:
            st.error("❌ Failed to decode image. Please upload a valid image file.")
        else:
            original_img = cv2.resize(original_img, (224, 224))
            
            # Reset pointer for preprocessing
            uploaded_file.seek(0)
            
            # --------------------------------------------------
            # PREPROCESS
            # --------------------------------------------------
            with st.spinner("🔄 Preprocessing image..."):
                preprocessed_img = preprocess_uploaded_image(uploaded_file)
            
            # --------------------------------------------------
            # PREDICTION
            # --------------------------------------------------
            with st.spinner("🤖 Analyzing MRI scan..."):
                predicted_class, confidence = predict_dementia(model, preprocessed_img)
                probabilities = get_prediction_probabilities(model, preprocessed_img)
            
            # --------------------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------------------
            st.subheader("🖼 Uploaded MRI Image")
            st.image(original_img, channels="BGR", use_container_width=True)
            
            st.divider()
            
            # --------------------------------------------------
            # PREDICTION DETAILS
            # --------------------------------------------------
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Prediction Result")
                # Color code based on prediction
                if predicted_class == "NonDemented":
                    st.success(f"**Predicted Stage:** {predicted_class} ✅")
                elif predicted_class == "VeryMildDemented":
                    st.warning(f"**Predicted Stage:** {predicted_class} ⚠️")
                elif predicted_class == "MildDemented":
                    st.error(f"**Predicted Stage:** {predicted_class} ⚠️")
                else:  # ModerateDemented
                    st.error(f"**Predicted Stage:** {predicted_class} 🚨")
                
                st.info(f"**Confidence:** {confidence:.2f}%")
            
            with col2:
                st.subheader("📈 Class Probabilities")
                st.bar_chart(probabilities)
            
            # --------------------------------------------------
            # REPORT GENERATION
            # --------------------------------------------------
            st.divider()
            st.subheader("📄 Generate Medical Report")
            
            with st.spinner("📝 Generating medical report..."):
                try:
                    report_path = generate_report(
                        prediction=predicted_class,
                        confidence=confidence,
                        probabilities=probabilities,
                        patient_name=patient_name if patient_name else "Not Provided",
                        age=str(age) if age > 0 else "Not Provided",
                        gender=gender,
                        clinical_notes=clinical_notes if clinical_notes else "None provided"
                    )
                    
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Diagnostic Report (PDF)",
                            data=f,
                            file_name="Dementia_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"❌ Report generation failed: {str(e)}")
    
    else:
        st.info("⬆️ Please upload an MRI image to begin analysis.")

# --------------------------------------------------
# ROUTING
# --------------------------------------------------
if not st.session_state.authenticated:
    login_page()
else:
    main_app()
