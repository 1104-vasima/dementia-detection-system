<<<<<<< HEAD
# 🧠 Dementia Diagnosis System

A comprehensive Streamlit web application for diagnosing dementia from brain MRI images using deep learning, with AI-generated medical reports and Grad-CAM explainability.

## Features

### 🔐 Authentication Module
- Secure login system with username/password authentication
- Session state management
- Multiple user support

### 🧑‍⚕️ Patient Information Module
- Collect patient details (Name, Age, Gender, Clinical Notes)
- Input validation for mandatory fields
- Data persistence during session

### 🧠 MRI Image Upload & Display
- Support for JPG, JPEG, PNG formats
- Clear image display in the UI
- Real-time image preview

### ⚙️ Image Preprocessing Pipeline
- Automatic resizing to 224×224 pixels
- Pixel normalization to [0,1] range
- Dimension expansion for model inference
- Modular and reusable preprocessing functions

### 🤖 Dementia Classification Model
- Loads pretrained CNN/VGG/DenseNet model from `model.h5`
- Predicts one of four classes:
  - **NonDemented** (Green indicator)
  - **VeryMildDemented** (Orange indicator)
  - **MildDemented** (Orange indicator)
  - **ModerateDemented** (Red indicator)
- Displays predicted class and confidence score (%)

### 🔍 Grad-CAM Visualization (Explainable AI)
- Implements Grad-CAM++ visualization
- Overlays heatmap on original MRI image
- Highlights brain regions contributing to model decision
- Side-by-side comparison of original and Grad-CAM images

### 🧾 GenAI-Based Medical Report Generation
- Generates comprehensive clinical-style diagnostic reports
- Includes:
  - Patient details summary
  - Model prediction and confidence
  - Grad-CAM findings interpretation
  - Clinical recommendations
  - Important medical disclaimers
- Professional, medical, and academic tone
- Supports OpenAI API integration (optional)

### 📥 Downloadable Reports
- Generate and download:
  - **Text Report** (.txt format)
  - **PDF Report** (.pdf format) with embedded images
- Reports include:
  - Uploaded MRI image
  - Grad-CAM heatmap visualization
  - Prediction results
  - Generated AI explanation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
Ensure all files are in your project directory:
```
dementia/
├── app.py
├── requirements.txt
├── model.h5 (your pretrained model)
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model_loader.py
│   ├── gradcam.py
│   └── report_generator.py
└── README.md
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add Your Model
Place your pretrained model file (`model.h5`) in the root directory of the project.

**Note:** If `model.h5` is not found, the application will use a dummy model for demonstration purposes. For actual predictions, ensure your model file is present.

### Step 4: (Optional) Configure OpenAI API
For enhanced AI-generated reports, set your OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

If no API key is provided, the system will use template-based reports.

## Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Login Credentials
The application comes with demo credentials:
- **Username:** `admin` | **Password:** `admin123`
- **Username:** `doctor` | **Password:** `doctor123`
- **Username:** `user` | **Password:** `user123`

### Workflow

1. **Login**: Enter your credentials on the login page

2. **Enter Patient Information**:
   - Fill in patient name (required)
   - Enter age (required)
   - Select gender (required)
   - Add clinical notes/symptoms (optional)
   - Click "Save Patient Information"

3. **Upload MRI Image**:
   - Click "Browse files" or drag and drop an MRI image
   - Supported formats: JPG, JPEG, PNG
   - Image will be displayed automatically

4. **Analyze**:
   - Click "🔍 Analyze MRI Scan" button
   - Wait for processing (image preprocessing, model prediction, Grad-CAM generation)

5. **View Results**:
   - See predicted class with color-coded indicator
   - View confidence score
   - Examine Grad-CAM visualization

6. **Generate Report**:
   - Click "Generate Medical Report"
   - Review the comprehensive AI-generated report
   - Download as TXT or PDF

7. **Logout**: Click "Logout" button when finished

## Model Requirements

Your `model.h5` file should:
- Be a Keras/TensorFlow model
- Accept input shape: `(None, 224, 224, 3)`
- Output 4 classes (or be adaptable to 4 classes)
- Use softmax activation for the output layer

### Expected Output Classes:
- Class 0: NonDemented
- Class 1: VeryMildDemented
- Class 2: MildDemented
- Class 3: ModerateDemented

## Project Structure

```
dementia/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── model.h5                    # Pretrained model (user-provided)
├── utils/
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py       # Image preprocessing utilities
│   ├── model_loader.py        # Model loading and prediction
│   ├── gradcam.py             # Grad-CAM visualization
│   └── report_generator.py    # Medical report generation
└── README.md                   # This file
```

## Technical Details

### Image Preprocessing
- **Resize**: All images are resized to 224×224 pixels using LANCZOS resampling
- **Normalization**: Pixel values normalized to [0, 1] range
- **Format**: Images converted to RGB if needed
- **Dimensions**: Expanded to (1, 224, 224, 3) for model input

### Grad-CAM Implementation
- Automatically detects the last convolutional layer
- Computes gradient-weighted class activation maps
- Overlays heatmap with configurable transparency (default: 40%)
- Uses JET colormap for visualization

### Report Generation
- **Template-based**: Always available, generates professional medical reports
- **AI-enhanced**: Uses OpenAI GPT-3.5-turbo when API key is available
- **PDF Generation**: Uses ReportLab library for PDF creation with embedded images

## Troubleshooting

### Model Not Found
- Ensure `model.h5` is in the root directory
- Check file permissions
- The app will use a dummy model for demonstration if model is missing

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Grad-CAM Not Working
- Ensure your model has convolutional layers
- Check that the model architecture is compatible
- The app will fall back to original image if Grad-CAM fails

### PDF Generation Issues
- Ensure `reportlab` is installed: `pip install reportlab`
- Check that images are in valid format

### OpenAI API Issues
- Verify API key is set correctly
- Check internet connection
- App will fall back to template-based reports if API fails

## Security Notes

⚠️ **Important Security Considerations:**

1. **Hardcoded Credentials**: The current implementation uses hardcoded credentials for demonstration. For production use:
   - Implement proper authentication (OAuth, JWT, etc.)
   - Use environment variables or secure credential storage
   - Implement password hashing

2. **Model Security**: Ensure your model file is from a trusted source

3. **Data Privacy**: Patient data is stored in session state only. For production:
   - Implement proper database storage
   - Ensure HIPAA compliance
   - Implement data encryption

4. **API Keys**: Never commit API keys to version control

## Medical Disclaimer

⚠️ **CRITICAL MEDICAL DISCLAIMER:**

This application is for **research and educational purposes only**. It is NOT a medical device and should NOT be used for actual clinical diagnosis. All predictions and reports are AI-assisted and must be reviewed by qualified healthcare professionals. The developers assume no responsibility for any medical decisions made based on this software.

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

---

**Built with ❤️ using Streamlit, TensorFlow, and modern AI technologies**
=======
# 🧠 Dementia Diagnosis System

A comprehensive Streamlit web application for diagnosing dementia from brain MRI images using deep learning, with AI-generated medical reports and Grad-CAM explainability.

## Features

### 🔐 Authentication Module
- Secure login system with username/password authentication
- Session state management
- Multiple user support

### 🧑‍⚕️ Patient Information Module
- Collect patient details (Name, Age, Gender, Clinical Notes)
- Input validation for mandatory fields
- Data persistence during session

### 🧠 MRI Image Upload & Display
- Support for JPG, JPEG, PNG formats
- Clear image display in the UI
- Real-time image preview

### ⚙️ Image Preprocessing Pipeline
- Automatic resizing to 224×224 pixels
- Pixel normalization to [0,1] range
- Dimension expansion for model inference
- Modular and reusable preprocessing functions

### 🤖 Dementia Classification Model
- Loads pretrained CNN/VGG/DenseNet model from `model.h5`
- Predicts one of four classes:
  - **NonDemented** (Green indicator)
  - **VeryMildDemented** (Orange indicator)
  - **MildDemented** (Orange indicator)
  - **ModerateDemented** (Red indicator)
- Displays predicted class and confidence score (%)

### 🔍 Grad-CAM Visualization (Explainable AI)
- Implements Grad-CAM++ visualization
- Overlays heatmap on original MRI image
- Highlights brain regions contributing to model decision
- Side-by-side comparison of original and Grad-CAM images

### 🧾 GenAI-Based Medical Report Generation
- Generates comprehensive clinical-style diagnostic reports
- Includes:
  - Patient details summary
  - Model prediction and confidence
  - Grad-CAM findings interpretation
  - Clinical recommendations
  - Important medical disclaimers
- Professional, medical, and academic tone
- Supports OpenAI API integration (optional)

### 📥 Downloadable Reports
- Generate and download:
  - **Text Report** (.txt format)
  - **PDF Report** (.pdf format) with embedded images
- Reports include:
  - Uploaded MRI image
  - Grad-CAM heatmap visualization
  - Prediction results
  - Generated AI explanation

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
Ensure all files are in your project directory:
```
dementia/
├── app.py
├── requirements.txt
├── model.h5 (your pretrained model)
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model_loader.py
│   ├── gradcam.py
│   └── report_generator.py
└── README.md
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add Your Model
Place your pretrained model file (`model.h5`) in the root directory of the project.

**Note:** If `model.h5` is not found, the application will use a dummy model for demonstration purposes. For actual predictions, ensure your model file is present.

### Step 4: (Optional) Configure OpenAI API
For enhanced AI-generated reports, set your OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

If no API key is provided, the system will use template-based reports.

## Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Login Credentials
The application comes with demo credentials:
- **Username:** `admin` | **Password:** `admin123`
- **Username:** `doctor` | **Password:** `doctor123`
- **Username:** `user` | **Password:** `user123`

### Workflow

1. **Login**: Enter your credentials on the login page

2. **Enter Patient Information**:
   - Fill in patient name (required)
   - Enter age (required)
   - Select gender (required)
   - Add clinical notes/symptoms (optional)
   - Click "Save Patient Information"

3. **Upload MRI Image**:
   - Click "Browse files" or drag and drop an MRI image
   - Supported formats: JPG, JPEG, PNG
   - Image will be displayed automatically

4. **Analyze**:
   - Click "🔍 Analyze MRI Scan" button
   - Wait for processing (image preprocessing, model prediction, Grad-CAM generation)

5. **View Results**:
   - See predicted class with color-coded indicator
   - View confidence score
   - Examine Grad-CAM visualization

6. **Generate Report**:
   - Click "Generate Medical Report"
   - Review the comprehensive AI-generated report
   - Download as TXT or PDF

7. **Logout**: Click "Logout" button when finished

## Model Requirements

Your `model.h5` file should:
- Be a Keras/TensorFlow model
- Accept input shape: `(None, 224, 224, 3)`
- Output 4 classes (or be adaptable to 4 classes)
- Use softmax activation for the output layer

### Expected Output Classes:
- Class 0: NonDemented
- Class 1: VeryMildDemented
- Class 2: MildDemented
- Class 3: ModerateDemented

## Project Structure

```
dementia/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── model.h5                    # Pretrained model (user-provided)
├── utils/
│   ├── __init__.py            # Package initialization
│   ├── preprocessing.py       # Image preprocessing utilities
│   ├── model_loader.py        # Model loading and prediction
│   ├── gradcam.py             # Grad-CAM visualization
│   └── report_generator.py    # Medical report generation
└── README.md                   # This file
```

## Technical Details

### Image Preprocessing
- **Resize**: All images are resized to 224×224 pixels using LANCZOS resampling
- **Normalization**: Pixel values normalized to [0, 1] range
- **Format**: Images converted to RGB if needed
- **Dimensions**: Expanded to (1, 224, 224, 3) for model input

### Grad-CAM Implementation
- Automatically detects the last convolutional layer
- Computes gradient-weighted class activation maps
- Overlays heatmap with configurable transparency (default: 40%)
- Uses JET colormap for visualization

### Report Generation
- **Template-based**: Always available, generates professional medical reports
- **AI-enhanced**: Uses OpenAI GPT-3.5-turbo when API key is available
- **PDF Generation**: Uses ReportLab library for PDF creation with embedded images

## Troubleshooting

### Model Not Found
- Ensure `model.h5` is in the root directory
- Check file permissions
- The app will use a dummy model for demonstration if model is missing

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Grad-CAM Not Working
- Ensure your model has convolutional layers
- Check that the model architecture is compatible
- The app will fall back to original image if Grad-CAM fails

### PDF Generation Issues
- Ensure `reportlab` is installed: `pip install reportlab`
- Check that images are in valid format

### OpenAI API Issues
- Verify API key is set correctly
- Check internet connection
- App will fall back to template-based reports if API fails

## Security Notes

⚠️ **Important Security Considerations:**

1. **Hardcoded Credentials**: The current implementation uses hardcoded credentials for demonstration. For production use:
   - Implement proper authentication (OAuth, JWT, etc.)
   - Use environment variables or secure credential storage
   - Implement password hashing

2. **Model Security**: Ensure your model file is from a trusted source

3. **Data Privacy**: Patient data is stored in session state only. For production:
   - Implement proper database storage
   - Ensure HIPAA compliance
   - Implement data encryption

4. **API Keys**: Never commit API keys to version control

## Medical Disclaimer

⚠️ **CRITICAL MEDICAL DISCLAIMER:**

This application is for **research and educational purposes only**. It is NOT a medical device and should NOT be used for actual clinical diagnosis. All predictions and reports are AI-assisted and must be reviewed by qualified healthcare professionals. The developers assume no responsibility for any medical decisions made based on this software.

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or contributions, please refer to the project repository or contact the development team.

---

**Built with ❤️ using Streamlit, TensorFlow, and modern AI technologies**
>>>>>>> 29fa703cec59b2eec93aedccc891802ebc584cbc
