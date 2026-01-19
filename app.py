import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
from datetime import datetime
import base64

from utils.preprocessing import preprocess_image
from utils.model_loader import load_model, predict_dementia
from utils.gradcam import generate_gradcam
from utils.report_generator import generate_medical_report

# Page configuration
st.set_page_config(
    page_title="Dementia Diagnosis System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .non-demented {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .demented {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .mild-demented {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Hardcoded credentials (for demo purposes)
VALID_CREDENTIALS = {
    "admin": "admin123",
    "doctor": "doctor123",
    "user": "user123"
}

def check_authentication():
    """Check if user is authenticated"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    return st.session_state.authenticated

def login_page():
    """Display login page"""
    st.markdown('<div class="main-header">🧠 Dementia Diagnosis System</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        st.markdown("---")
        st.info("**Demo Credentials:**\n- Username: admin, Password: admin123\n- Username: doctor, Password: doctor123\n- Username: user, Password: user123")

def main_dashboard():
    """Main dashboard after authentication"""
    # Header with logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">🧠 Dementia Diagnosis System</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.patient_data = {}
            st.session_state.uploaded_image = None
            st.session_state.prediction = None
            st.rerun()
    
    st.markdown("---")
    
    # Patient Information Section
    st.markdown("## 📋 Patient Information")
    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name *", value=st.session_state.get('patient_name', ''))
            age = st.number_input("Age *", min_value=1, max_value=120, value=st.session_state.get('age', 50))
        with col2:
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"], index=0 if st.session_state.get('gender') != "Female" else 1)
            clinical_notes = st.text_area("Clinical Notes / Symptoms", value=st.session_state.get('clinical_notes', ''), height=100)
        
        submitted = st.form_submit_button("Save Patient Information", type="primary")
        
        if submitted:
            if not patient_name or not age:
                st.error("Please fill in all mandatory fields (marked with *)")
            else:
                st.session_state.patient_name = patient_name
                st.session_state.age = age
                st.session_state.gender = gender
                st.session_state.clinical_notes = clinical_notes
                st.success("Patient information saved successfully!")
    
    st.markdown("---")
    
    # MRI Image Upload Section
    st.markdown("## 🧠 MRI Brain Scan Upload")
    uploaded_file = st.file_uploader(
        "Upload MRI Brain Scan Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI image in JPG, JPEG, or PNG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.session_state.uploaded_file = uploaded_file
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Uploaded MRI Image")
            st.image(image, caption="Original MRI Scan", use_container_width=True)
        
        # Process and predict
        if st.button("🔍 Analyze MRI Scan", type="primary", use_container_width=True):
            with st.spinner("Processing image and running diagnosis..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Load model and predict
                    model = load_model()
                    prediction, confidence = predict_dementia(model, processed_image)
                    
                    # Store prediction
                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.session_state.processed_image = processed_image
                    
                    # Generate Grad-CAM
                    with st.spinner("Generating Grad-CAM visualization..."):
                        gradcam_image = generate_gradcam(model, processed_image, image)
                        st.session_state.gradcam_image = gradcam_image
                    
                    st.success("Analysis complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
        
        # Display results if prediction exists
        if 'prediction' in st.session_state and st.session_state.prediction is not None:
            st.markdown("---")
            st.markdown("## 📊 Diagnosis Results")
            
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            
            # Determine color class
            if prediction == "NonDemented":
                color_class = "non-demented"
                color = "green"
                emoji = "✅"
            elif prediction == "VeryMildDemented":
                color_class = "mild-demented"
                color = "orange"
                emoji = "⚠️"
            elif prediction == "MildDemented":
                color_class = "mild-demented"
                color = "orange"
                emoji = "⚠️"
            else:  # ModerateDemented
                color_class = "demented"
                color = "red"
                emoji = "🔴"
            
            # Display prediction
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                    <div class="prediction-box {color_class}">
                        <h2>{emoji} Predicted Class: {prediction}</h2>
                        <h3>Confidence: {confidence:.2f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Prediction", prediction)
                st.metric("Confidence", f"{confidence:.2f}%")
            
            # Display Grad-CAM visualization
            if 'gradcam_image' in st.session_state:
                st.markdown("---")
                st.markdown("## 🔍 Grad-CAM Visualization (Explainable AI)")
                st.markdown("**Heatmap highlighting brain regions contributing to the diagnosis:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state.uploaded_image, caption="Original MRI", use_container_width=True)
                with col2:
                    st.image(st.session_state.gradcam_image, caption="Grad-CAM Heatmap", use_container_width=True)
            
            # Generate and display medical report
            st.markdown("---")
            st.markdown("## 🧾 AI-Generated Medical Report")
            
            if st.button("Generate Medical Report", type="primary"):
                with st.spinner("Generating comprehensive medical report using AI..."):
                    try:
                        report = generate_medical_report(
                            patient_name=st.session_state.get('patient_name', 'N/A'),
                            age=st.session_state.get('age', 'N/A'),
                            gender=st.session_state.get('gender', 'N/A'),
                            clinical_notes=st.session_state.get('clinical_notes', 'None'),
                            prediction=prediction,
                            confidence=confidence
                        )
                        st.session_state.medical_report = report
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
            
            if 'medical_report' in st.session_state:
                st.text_area("Medical Report", st.session_state.medical_report, height=400)
                
                # Download buttons
                st.markdown("---")
                st.markdown("## 📥 Download Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download text report
                    report_text = st.session_state.medical_report
                    st.download_button(
                        label="📄 Download Text Report",
                        data=report_text,
                        file_name=f"dementia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Download PDF report (if reportlab is available)
                    try:
                        from utils.report_generator import generate_pdf_report
                        pdf_buffer = generate_pdf_report(
                            patient_name=st.session_state.get('patient_name', 'N/A'),
                            age=st.session_state.get('age', 'N/A'),
                            gender=st.session_state.get('gender', 'N/A'),
                            clinical_notes=st.session_state.get('clinical_notes', 'None'),
                            prediction=prediction,
                            confidence=confidence,
                            original_image=st.session_state.uploaded_image,
                            gradcam_image=st.session_state.gradcam_image if 'gradcam_image' in st.session_state else None,
                            report_text=st.session_state.medical_report
                        )
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"dementia_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except ImportError:
                        st.info("PDF generation requires reportlab. Install with: pip install reportlab")
                    except Exception as e:
                        st.warning(f"PDF generation error: {str(e)}")

def main():
    """Main application entry point"""
    if not check_authentication():
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()
