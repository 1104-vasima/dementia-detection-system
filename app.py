import streamlit as st
import numpy as np
import cv2

from utils.model_loader import load_model, predict_dementia, get_prediction_probabilities
from utils.preprocessing import preprocess_uploaded_image
from utils.gradcam import generate_gradcam, overlay_gradcam
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
# AUTHENTICATION
# --------------------------------------------------
# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

# User credentials (in production, use a database or secure storage)
USERS = {
    "admin": "admin123",
    "doctor": "doctor123",
    "user": "user123"
}

def login_page():
    """Display login page"""
    st.title("🧠 Dementia Detection System")
    st.markdown("### Please login to continue")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username in USERS and USERS[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"✅ Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        st.markdown("---")
        st.info("**Demo Credentials:**\n- Username: `admin` | Password: `admin123`\n- Username: `doctor` | Password: `doctor123`\n- Username: `user` | Password: `user123`")

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
            # GRADCAM
            # --------------------------------------------------
            with st.spinner("🔥 Generating Grad-CAM visualization..."):
                try:
                    heatmap = generate_gradcam(model, preprocessed_img)
                    # Use improved overlay with better alpha and colormap
                    gradcam_img = overlay_gradcam(original_img, heatmap, alpha=0.5, colormap_type='jet')
                except Exception as e:
                    st.error(f"❌ Grad-CAM generation failed: {str(e)}")
                    gradcam_img = original_img.copy()
                    heatmap = None
            
            # --------------------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------------------
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🖼 Uploaded MRI Image")
                st.image(original_img, channels="BGR", use_container_width=True)
            
            with col2:
                st.subheader("🔥 Grad-CAM Explanation")
                if heatmap is not None:
                    st.image(gradcam_img, channels="BGR", use_container_width=True)
                    st.caption("Red regions indicate areas most important for the prediction")
                else:
                    st.image(original_img, channels="BGR", use_container_width=True)
                    st.caption("Grad-CAM visualization unavailable")
            
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
                        gradcam_image=gradcam_img,
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
