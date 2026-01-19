"""
Medical report generation using Generative AI
"""
import os
from datetime import datetime
import io
from PIL import Image

def generate_medical_report(patient_name, age, gender, clinical_notes, prediction, confidence):
    """
    Generate a comprehensive medical report using AI
    
    Args:
        patient_name: Patient's name
        age: Patient's age
        gender: Patient's gender
        clinical_notes: Clinical notes/symptoms
        prediction: Predicted dementia class
        confidence: Confidence score (0-100)
        
    Returns:
        Generated medical report as string
    """
    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        try:
            return generate_report_with_openai(
                patient_name, age, gender, clinical_notes, prediction, confidence, openai_api_key
            )
        except Exception as e:
            # Fallback to template-based report if API fails
            return generate_template_report(
                patient_name, age, gender, clinical_notes, prediction, confidence
            )
    else:
        # Use template-based report if no API key
        return generate_template_report(
            patient_name, age, gender, clinical_notes, prediction, confidence
        )

def generate_report_with_openai(patient_name, age, gender, clinical_notes, prediction, confidence, api_key):
    """
    Generate report using OpenAI API
    """
    try:
        import openai
        
        openai.api_key = api_key
        
        prompt = f"""Generate a professional medical diagnostic report for a dementia assessment based on brain MRI analysis.

Patient Information:
- Name: {patient_name}
- Age: {age}
- Gender: {gender}
- Clinical Notes: {clinical_notes}

AI Model Analysis:
- Predicted Class: {prediction}
- Confidence Level: {confidence:.2f}%

Please generate a comprehensive medical report that includes:
1. Patient information summary
2. Clinical presentation
3. AI model findings and interpretation
4. Grad-CAM visualization interpretation (highlighting brain regions of interest)
5. Clinical recommendations
6. Important disclaimers about AI-assisted diagnosis

The report should be professional, medical, and academic in tone. Include appropriate medical terminology and maintain clinical objectivity."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical professional writing diagnostic reports. Be professional, accurate, and include appropriate medical disclaimers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        raise Exception("OpenAI library not installed. Install with: pip install openai")
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def generate_template_report(patient_name, age, gender, clinical_notes, prediction, confidence):
    """
    Generate a template-based medical report (fallback when AI is not available)
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Determine severity interpretation
    if prediction == "NonDemented":
        severity = "No significant signs of dementia detected"
        recommendation = "Continue routine monitoring. No immediate intervention required."
        urgency = "Low"
    elif prediction == "VeryMildDemented":
        severity = "Very mild cognitive impairment detected"
        recommendation = "Recommend follow-up neuropsychological assessment and monitoring. Consider lifestyle interventions."
        urgency = "Moderate"
    elif prediction == "MildDemented":
        severity = "Mild dementia detected"
        recommendation = "Immediate clinical evaluation recommended. Consider comprehensive cognitive assessment and potential treatment options."
        urgency = "High"
    else:  # ModerateDemented
        severity = "Moderate dementia detected"
        recommendation = "Urgent clinical evaluation required. Immediate intervention and treatment planning recommended."
        urgency = "Critical"
    
    report = f"""
================================================================================
                    DEMENTIA DIAGNOSTIC REPORT
                    AI-Assisted Brain MRI Analysis
================================================================================

Report Date: {current_date}
Report Time: {current_time}
Report Type: AI-Assisted Diagnostic Analysis

--------------------------------------------------------------------------------
PATIENT INFORMATION
--------------------------------------------------------------------------------

Patient Name: {patient_name}
Age: {age} years
Gender: {gender}
Clinical Notes/Symptoms: {clinical_notes if clinical_notes else "None provided"}

--------------------------------------------------------------------------------
AI MODEL ANALYSIS RESULTS
--------------------------------------------------------------------------------

Predicted Classification: {prediction}
Confidence Level: {confidence:.2f}%
Severity Assessment: {severity}

Model Performance:
The deep learning model analyzed the brain MRI scan using advanced convolutional 
neural network architecture. The model has been trained on a dataset of brain MRI 
images with corresponding dementia classifications.

Confidence Interpretation:
- Confidence ≥ 80%: High confidence in prediction
- Confidence 60-79%: Moderate confidence
- Confidence < 60%: Lower confidence, additional validation recommended

Current Prediction Confidence: {confidence:.2f}%

--------------------------------------------------------------------------------
GRAD-CAM VISUALIZATION INTERPRETATION
--------------------------------------------------------------------------------

The Grad-CAM (Gradient-weighted Class Activation Mapping) visualization highlights 
brain regions that contributed most significantly to the model's classification 
decision. The heatmap overlay on the MRI scan shows:

1. Red/Hot Regions: Areas with highest activation, indicating regions the model 
   identified as most relevant for the dementia classification.

2. Blue/Cold Regions: Areas with lower activation, indicating regions with less 
   influence on the classification decision.

3. Clinical Significance: The highlighted regions typically correspond to areas 
   commonly associated with dementia-related changes, including:
   - Hippocampal regions
   - Temporal lobe structures
   - Cortical atrophy patterns
   - Ventricular enlargement

Note: The Grad-CAM visualization provides insight into model decision-making but 
should be interpreted in conjunction with clinical expertise and additional 
diagnostic modalities.

--------------------------------------------------------------------------------
CLINICAL INTERPRETATION
--------------------------------------------------------------------------------

Based on the AI model analysis:

Classification: {prediction}
Severity Level: {severity}
Clinical Urgency: {urgency}

The model's prediction suggests {prediction.lower()} classification with a 
confidence of {confidence:.2f}%. This assessment is based on pattern recognition 
in the brain MRI scan, comparing the patient's scan against learned patterns from 
the training dataset.

Key Observations:
- The AI model has identified features consistent with {prediction.lower()} 
  classification
- Confidence level indicates {'high' if confidence >= 80 else 'moderate' if confidence >= 60 else 'moderate to low'} 
  reliability in this prediction
- Grad-CAM visualization highlights specific brain regions of interest

--------------------------------------------------------------------------------
CLINICAL RECOMMENDATIONS
--------------------------------------------------------------------------------

{recommendation}

Additional Recommendations:
1. Clinical Correlation: This AI-assisted diagnosis should be correlated with 
   comprehensive clinical evaluation, including:
   - Detailed patient history
   - Neuropsychological testing
   - Laboratory studies
   - Additional imaging if indicated

2. Follow-up: Schedule appropriate follow-up based on clinical urgency ({urgency})

3. Multidisciplinary Approach: Consider consultation with:
   - Neurologist
   - Neuropsychologist
   - Geriatrician (if applicable)

4. Patient and Family Counseling: Provide appropriate counseling regarding 
   findings and next steps

--------------------------------------------------------------------------------
IMPORTANT DISCLAIMERS
--------------------------------------------------------------------------------

⚠️  CRITICAL DISCLAIMERS:

1. AI-ASSISTED DIAGNOSIS: This report is generated using artificial intelligence 
   and machine learning algorithms. It is NOT a substitute for professional 
   medical judgment, clinical evaluation, or comprehensive diagnostic workup.

2. NOT A FINAL DIAGNOSIS: This AI-assisted analysis should be considered as a 
   supplementary tool to assist healthcare professionals. It does not constitute 
   a final diagnosis.

3. CLINICAL CORRELATION REQUIRED: All findings must be correlated with:
   - Clinical presentation
   - Patient history
   - Physical examination
   - Additional diagnostic tests
   - Professional medical judgment

4. LIMITATIONS: AI models have limitations including:
   - Potential for false positives and false negatives
   - Dependence on image quality and preprocessing
   - Training data limitations
   - Lack of clinical context integration

5. RESPONSIBILITY: The ultimate responsibility for diagnosis and treatment 
   decisions rests with qualified healthcare professionals, not with the AI 
   system or this automated report.

6. REGULATORY STATUS: This system is for research and educational purposes. 
   Clinical use should comply with applicable medical device regulations and 
   institutional policies.

7. PATIENT PRIVACY: All patient information should be handled in accordance 
   with HIPAA and applicable privacy regulations.

--------------------------------------------------------------------------------
REPORT GENERATION INFORMATION
--------------------------------------------------------------------------------

Report Generated By: AI-Assisted Dementia Diagnosis System
Model Version: Deep Learning CNN/VGG/DenseNet Architecture
Analysis Method: Convolutional Neural Network with Grad-CAM Explainability
Report Type: Automated AI-Generated Diagnostic Report

================================================================================
END OF REPORT
================================================================================

For questions or concerns regarding this report, please consult with a qualified 
healthcare professional.

This report is confidential and intended solely for the use of the patient and 
their healthcare providers.
"""
    
    return report

def generate_pdf_report(patient_name, age, gender, clinical_notes, prediction, confidence, 
                       original_image, gradcam_image, report_text):
    """
    Generate a PDF report with images embedded
    
    Args:
        patient_name: Patient's name
        age: Patient's age
        gender: Patient's gender
        clinical_notes: Clinical notes
        prediction: Predicted class
        confidence: Confidence score
        original_image: PIL Image of original MRI
        gradcam_image: PIL Image of Grad-CAM visualization
        report_text: Text content of the report
        
    Returns:
        BytesIO buffer containing PDF
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Container for the 'Flowable' objects
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph("DEMENTIA DIAGNOSTIC REPORT", title_style))
        elements.append(Paragraph("AI-Assisted Brain MRI Analysis", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Patient Information
        elements.append(Paragraph("PATIENT INFORMATION", heading_style))
        elements.append(Paragraph(f"<b>Name:</b> {patient_name}", styles['Normal']))
        elements.append(Paragraph(f"<b>Age:</b> {age} years", styles['Normal']))
        elements.append(Paragraph(f"<b>Gender:</b> {gender}", styles['Normal']))
        elements.append(Paragraph(f"<b>Clinical Notes:</b> {clinical_notes if clinical_notes else 'None'}", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Prediction Results
        elements.append(Paragraph("AI MODEL ANALYSIS RESULTS", heading_style))
        elements.append(Paragraph(f"<b>Predicted Classification:</b> {prediction}", styles['Normal']))
        elements.append(Paragraph(f"<b>Confidence Level:</b> {confidence:.2f}%", styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Images
        if original_image:
            # Save images to temporary buffers
            img_buffer_orig = io.BytesIO()
            original_image.save(img_buffer_orig, format='PNG')
            img_buffer_orig.seek(0)
            
            elements.append(Paragraph("ORIGINAL MRI SCAN", heading_style))
            img_orig = RLImage(img_buffer_orig, width=4*inch, height=4*inch)
            elements.append(img_orig)
            elements.append(Spacer(1, 0.2*inch))
        
        if gradcam_image:
            img_buffer_grad = io.BytesIO()
            gradcam_image.save(img_buffer_grad, format='PNG')
            img_buffer_grad.seek(0)
            
            elements.append(Paragraph("GRAD-CAM VISUALIZATION", heading_style))
            img_grad = RLImage(img_buffer_grad, width=4*inch, height=4*inch)
            elements.append(img_grad)
            elements.append(Spacer(1, 0.2*inch))
        
        # Report Text
        elements.append(Paragraph("COMPREHENSIVE REPORT", heading_style))
        # Split report text into paragraphs
        for paragraph in report_text.split('\n\n'):
            if paragraph.strip():
                elements.append(Paragraph(paragraph.strip().replace('\n', '<br/>'), styles['Normal']))
                elements.append(Spacer(1, 0.1*inch))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        raise ImportError("reportlab library not installed. Install with: pip install reportlab")
    except Exception as e:
        raise Exception(f"PDF generation error: {str(e)}")
