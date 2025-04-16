# Import required libraries
import streamlit as st  # For creating the web interface
import pandas as pd     # For data manipulation
import numpy as np      # For numerical operations
from datetime import datetime, timedelta  # For handling dates and times
import os              # For file operations
import sys
import logging
from prediction import predict  # Import the prediction function
import tempfile
from PIL import Image
import tensorflow as tf

# Add authentication and database modules
from auth import require_login, init_auth, is_admin, auth_page
from database.firebase_utils import save_prediction, save_appointment
import admin
import patient

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our image classification functions
# Define paths correctly regardless of where the script is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from predict import load_model as load_image_model
    from predict import predict as predict_image
    logger.info("Successfully imported image classification modules")
except ImportError as e:
    logger.error(f"Error importing image classification modules: {e}")
    
# Define skin cancer classes for image classification
SKIN_CLASSES = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Function to preprocess uploaded image
def preprocess_uploaded_image(uploaded_file, target_size=(75, 100)):
    """
    Preprocess an uploaded image file for prediction
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            # Write the uploaded file data to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Open the image
        img = Image.open(tmp_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Resize to target size - PIL uses (width, height) format
        img = img.resize((target_size[1], target_size[0]))
        
        # Save a copy of the processed image for display
        display_img = img.resize((300, 225))  # Larger size for display
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize image (similar to training)
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return img_array, display_img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None

# Function to load the image classification model
@st.cache_resource
def load_cached_image_model(model_path):
    """
    Load the image classification model with caching
    """
    try:
        model = load_image_model(model_path)
        logger.info(f"Image classification model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading image classification model: {e}")
        return None

# Configure the Streamlit page settings
st.set_page_config(
    page_title="Skin Cancer Classification",  # Title shown in browser tab
    page_icon="🔬",                          # Favicon
    layout="wide",                           # Use full width of the page
    initial_sidebar_state="expanded"         # Show sidebar by default
)

# Initialize authentication
init_auth()

# Initialize session state variables
# These variables persist across reruns of the app
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'image_model_loaded' not in st.session_state:
    st.session_state.image_model_loaded = False
if 'image_prediction_made' not in st.session_state:
    st.session_state.image_prediction_made = False
if 'image_prediction_results' not in st.session_state:
    st.session_state.image_prediction_results = None
if 'clinical_prediction_made' not in st.session_state:
    st.session_state.clinical_prediction_made = False
if 'clinical_prediction_results' not in st.session_state:
    st.session_state.clinical_prediction_results = None
if 'current_prediction_id' not in st.session_state:
    st.session_state.current_prediction_id = None
if 'temp_image_path' not in st.session_state:
    st.session_state.temp_image_path = None

# Initialize patient details with default Pakistani sample values
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = "Ali Ahmed Khan"
if 'patient_email' not in st.session_state:
    st.session_state.patient_email = "ali.khan@example.com"
if 'patient_phone' not in st.session_state:
    st.session_state.patient_phone = "+92 300 1234567"
if 'patient_address' not in st.session_state:
    st.session_state.patient_address = "House #123, Street 5\nGulberg, Lahore\nPunjab, Pakistan\n54000"

# Initialize clinical features with default values
if 'age' not in st.session_state:
    st.session_state.age = 45
if 'gender' not in st.session_state:
    st.session_state.gender = "MALE"
if 'fitzpatrick' not in st.session_state:
    st.session_state.fitzpatrick = 3
if 'location' not in st.session_state:
    st.session_state.location = "ARM"
if 'diameter_1' not in st.session_state:
    st.session_state.diameter_1 = 5.0
if 'diameter_2' not in st.session_state:
    st.session_state.diameter_2 = 5.0
if 'itching' not in st.session_state:
    st.session_state.itching = "No"
if 'growing' not in st.session_state:
    st.session_state.growing = "No"
if 'pain' not in st.session_state:
    st.session_state.pain = "No"
if 'bleeding' not in st.session_state:
    st.session_state.bleeding = "No"
if 'elevation' not in st.session_state:
    st.session_state.elevation = "No"
if 'smoking' not in st.session_state:
    st.session_state.smoking = "Never"
if 'cancer_history' not in st.session_state:
    st.session_state.cancer_history = "No"
if 'skin_cancer_history' not in st.session_state:
    st.session_state.skin_cancer_history = "No"
if 'pesticide_exposure' not in st.session_state:
    st.session_state.pesticide_exposure = "No"
if 'piped_water' not in st.session_state:
    st.session_state.piped_water = "Yes"
if 'sewage_system' not in st.session_state:
    st.session_state.sewage_system = "Yes"

# Function to load metadata for feature ranges
@st.cache_data  # Cache the result to improve performance
def load_metadata():
    try:
        # Try to load from web_app directory first
        try:
            df = pd.read_csv('metadata.csv')
        except:
            # If not found, try parent directory
            df = pd.read_csv('../metadata.csv')
        return df
    except Exception as e:
        st.error(f"Error loading metadata: {str(e)}")
        logger.error(f"Error loading metadata: {str(e)}")
        return None

# Function to prepare prediction input data
def prepare_prediction_input():
    """
    Prepare input data for model prediction from session state, aligning with expected model features.
    """
    # Map Yes/No to boolean values for categorical features
    def map_yes_no(value):
        return True if value == "Yes" else False
    
    # Convert smoking status to categories expected by model
    def map_smoking(value):
        if value == "Never":
            return "never"
        elif value == "Former":
            return "former"
        else: # Current
            return "current"
    
    # Create input data dictionary, matching expected keys from warnings
    input_data = {
        # Demographic features
        'age': st.session_state.age,
        'gender': st.session_state.gender,
        'fitspatrick': st.session_state.fitzpatrick, # Corrected key
        
        # Lesion features
        'region': st.session_state.location, # Renamed from 'location' to 'region'
        'diameter_1': st.session_state.diameter_1,
        'diameter_2': st.session_state.diameter_2,
        
        # Symptoms - matching expected keys
        'itch': map_yes_no(st.session_state.itching), # Renamed from 'itching'
        'grew': map_yes_no(st.session_state.growing), # Renamed from 'growing'
        'hurt': map_yes_no(st.session_state.pain),    # Renamed from 'pain'
        'bleed': map_yes_no(st.session_state.bleeding), # Renamed from 'bleeding'
        # 'changed': ??? # Not collected in UI
        # 'elevation': map_yes_no(st.session_state.elevation), # Model might not expect this one?
        
        # Medical history - matching expected keys
        'smoke': map_smoking(st.session_state.smoking), # Renamed from 'smoking'
        # 'drink': ??? # Not collected in UI
        # 'background_father': ??? # Not collected in UI
        # 'background_mother': ??? # Not collected in UI
        'cancer_history': map_yes_no(st.session_state.cancer_history),
        'skin_cancer_history': map_yes_no(st.session_state.skin_cancer_history),
        
        # Environmental factors - matching expected keys
        'pesticide': map_yes_no(st.session_state.pesticide_exposure), # Renamed
        'has_piped_water': map_yes_no(st.session_state.piped_water), # Renamed
        'has_sewage_system': map_yes_no(st.session_state.sewage_system) # Renamed
        # 'biopsed': ??? # Not collected in UI
    }
    
    logger.info(f"Prepared input data with keys: {list(input_data.keys())}")
    return input_data

# Main title and description of the application
st.title("🔬 Skin Cancer Classification System")
st.markdown("""
This application helps in preliminary skin cancer classification using either:
- Image analysis of skin lesions
- Clinical features and patient information
""")

# Check if user is authenticated
if not st.session_state.is_authenticated:
    # Show login/signup page
    auth_page()
    st.stop()  # This will prevent the rest of the app from loading

# If user is admin, redirect to admin page
if is_admin():
    admin.admin_dashboard()
    st.stop()

# Update patient information after login
if st.session_state.is_authenticated and st.session_state.user:
    # Update patient details from user profile
    st.session_state.patient_name = st.session_state.user.get('full_name', '')
    st.session_state.patient_email = st.session_state.user.get('email', '')
    st.session_state.patient_phone = st.session_state.user.get('phone', '')
    st.session_state.patient_address = st.session_state.user.get('address', '')

# Sidebar menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Skin Cancer Classification", "My Dashboard"])

if page == "My Dashboard":
    patient.patient_dashboard()
    st.stop()

# Otherwise, continue with the main app for classification

# Check if model and preprocessing objects are available
if not st.session_state.model_loaded:
    with st.spinner("Checking model availability..."):
        # Try a simple prediction to see if the model loads
        try:
            # Create a minimal input data dictionary with default values
            test_input = {
                'age': 45, 'gender': 'MALE', 'fitzpatrick': 3, 
                'location': 'ARM', 'diameter_1': 5.0, 'diameter_2': 5.0,
                'itching': False, 'growing': False, 'pain': False, 
                'bleeding': False, 'elevation': False, 'smoking': 'never',
                'cancer_history': False, 'skin_cancer_history': False,
                'pesticide_exposure': False, 'piped_water': True, 'sewage_system': True
            }
            
            # Try to make a prediction
            test_result = predict(test_input)
            
            # Check if prediction was successful
            if "error" in test_result and test_result["error"]:
                logger.error(f"Model check failed: {test_result['error']}")
                st.error(f"""
                ⚠️ **Model Not Available** ⚠️
                
                The trained model could not be loaded. Please make sure you have run `skin_cancer_model_clean.py` first to train the model.
                
                Error: {test_result['error']}
                """)
            else:
                st.session_state.model_loaded = True
                logger.info("Model successfully loaded")
        except Exception as e:
            logger.error(f"Error testing model availability: {str(e)}")
            st.error(f"""
            ⚠️ **Model Not Available** ⚠️
            
            There was an error loading the trained model: {str(e)}
            
            Please make sure you have run `skin_cancer_model_clean.py` first to train the model.
            """)

# Check if image classification model is available
if not st.session_state.image_model_loaded:
    with st.spinner("Checking image classification model availability..."):
        # Try to load the image model
        try:
            # Look for model in different possible locations
            model_paths = [
                'skin_cancer_model.keras',  # Current directory
                '../skin_cancer_model.keras',  # Parent directory
                os.path.join(parent_dir, 'skin_cancer_model.keras')  # Absolute path from parent dir
            ]
            
            image_model = None
            for path in model_paths:
                if os.path.exists(path):
                    logger.info(f"Found image model at {path}")
                    image_model = load_cached_image_model(path)
                    if image_model:
                        st.session_state.image_model_path = path
                        break
            
            if image_model:
                st.session_state.image_model_loaded = True
                logger.info("Image classification model successfully loaded")
            else:
                logger.warning("Image classification model not found")
        except Exception as e:
            logger.error(f"Error loading image classification model: {e}")

# Sidebar for input method selection
st.sidebar.title("Input Method")
input_method = st.sidebar.radio(
    "Choose how you want to provide information:",
    ["Image Upload", "Clinical Features"]
)

# Main content area - Image Upload Section
if input_method == "Image Upload":
    st.header("Image-based Classification")
    
    if st.session_state.image_model_loaded:
        st.markdown("Upload an image of the skin lesion for analysis")
    else:
        st.warning("⚠️ Image classification model not available. You can still use the Clinical Features method.")
        st.markdown("Upload an image of the skin lesion for visual reference only. Classification will not be available.")
    
    # Patient Details Section
    st.markdown("### Patient Details")
    st.markdown("""
    Please provide your basic contact information.
    """)
    
    # Input fields for patient information with helpful tooltips
    st.session_state.patient_name = st.text_input("Full Name", value=st.session_state.patient_name)
    st.info("Please enter your full legal name as it appears on your identification.")
    
    st.session_state.patient_email = st.text_input("Email Address", value=st.session_state.patient_email)
    st.info("We will use this email to send your appointment confirmation and results.")
    
    st.session_state.patient_phone = st.text_input("Phone Number", value=st.session_state.patient_phone)
    st.info("We may use this number to contact you regarding your appointment.")
    
    st.session_state.patient_address = st.text_area("Address", value=st.session_state.patient_address)
    st.info("Your complete address including city, state, and postal code.")
    
    # Image upload functionality
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        # Process image if model is available
        if st.session_state.image_model_loaded:
            # Button to trigger analysis
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Load model from session state
                        image_model = load_cached_image_model(st.session_state.image_model_path)
                        
                        # Preprocess the image
                        processed_image, display_img = preprocess_uploaded_image(uploaded_file)
                        
                        if processed_image is not None:
                            # Make prediction
                            result = predict_image(image_model, processed_image)
                            
                            if result:
                                # Map to expected format for displaying results
                                st.session_state.image_prediction_results = {
                                    "prediction": result["class_name"],
                                    "confidence": result["confidence"],
                                    "probabilities": result["all_probabilities"]
                                }
                                st.session_state.image_prediction_made = True
                                
                                # Display processed image
                                st.subheader("Processed Image")
                                st.image(display_img, width=300)
                                
                                # Save prediction to database
                                prediction_type = "Image Analysis"
                                prediction_result = result["class_name"]
                                confidence = result["confidence"]
                                
                                # Ensure confidence is a float 
                                try:
                                    confidence = float(confidence)
                                except (ValueError, TypeError):
                                    confidence = 0.0
                                
                                # Determine recommendation
                                if "melanoma" in prediction_result.lower():
                                    recommendation = "Urgent dermatologist consultation recommended"
                                elif "basal cell carcinoma" in prediction_result.lower() or "actinic keratoses" in prediction_result.lower():
                                    recommendation = "Dermatologist consultation recommended"
                                else:
                                    recommendation = "Regular follow-up recommended"
                                
                                # Save to database
                                pred_result = save_prediction(
                                    st.session_state.user['id'],
                                    prediction_type,
                                    prediction_result,
                                    confidence,
                                    recommendation,
                                    None  # No clinical features for image analysis
                                )
                                
                                if pred_result['success']:
                                    st.session_state.current_prediction_id = pred_result['prediction_id']
                                else:
                                    logger.error(f"Error saving prediction: {pred_result['error']}")
                                
                                logger.info(f"Image classification: {result['class_name']} with {result['confidence']:.2f}% confidence")
                            else:
                                st.error("Error during image analysis. Please try another image.")
                        else:
                            st.error("Error preprocessing image. Please try another image.")
                    except Exception as e:
                        st.error(f"Error during image analysis: {str(e)}")
                        logger.error(f"Error during image analysis: {str(e)}")
        else:
            st.warning("Image classification model is not loaded. Please upload the trained model file.")
            # Still store basic info for the rest of the app to work
            st.session_state.image_prediction_results = {
                "prediction": "Not available",
                "confidence": 0,
                "probabilities": {}
            }
            st.session_state.image_prediction_made = True

# Main content area - Clinical Features Section
else:  # Clinical Features
    st.header("Feature-based Classification")
    st.markdown("""
    Please provide the following clinical information. Each field includes additional details to help you provide accurate information.
    All measurements should be provided in metric units (millimeters, years) unless otherwise specified.
    """)
    
    # Load metadata for ranges
    df = load_metadata()
    if df is not None:
        # Patient Details Section
        st.markdown("### Patient Details")
        st.markdown("""
        Please provide your basic contact information.
        """)
        
        # Input fields for patient information
        st.session_state.patient_name = st.text_input("Full Name", value=st.session_state.patient_name)
        st.info("Please enter your full legal name as it appears on your identification.")
        
        st.session_state.patient_email = st.text_input("Email Address", value=st.session_state.patient_email)
        st.info("We will use this email to send your appointment confirmation and results.")
        
        st.session_state.patient_phone = st.text_input("Phone Number", value=st.session_state.patient_phone)
        st.info("We may use this number to contact you regarding your appointment.")
        
        st.session_state.patient_address = st.text_area("Address", value=st.session_state.patient_address)
        st.info("Your complete address including city, state, and postal code.")
        
        # Demographic Information Section
        st.markdown("### Demographic Information")
        st.markdown("""
        Basic demographic information helps us understand the patient's background and risk factors.
        """)
        
        # Input fields for demographic information
        st.session_state.age = st.slider("Age", 8, 79, st.session_state.age)
        st.info("Patient's current age in years. Age is an important factor in skin cancer risk assessment.")
        
        st.session_state.gender = st.radio("Gender", ["MALE", "FEMALE"], index=0 if st.session_state.gender == "MALE" else 1)
        st.info("Gender can influence the type and location of skin cancers.")
        
        # Use fitspatrick key consistent with model expectation
        st.session_state.fitzpatrick = st.select_slider("Fitzpatrick Skin Type", options=[1, 2, 3, 4, 5, 6], value=st.session_state.fitzpatrick)
        st.info("""
        Fitzpatrick Skin Type Classification (scale 1-6):
        - Type 1: Very fair skin, always burns, never tans
        - Type 2: Fair skin, burns easily, tans minimally
        - Type 3: Medium skin, sometimes burns, tans gradually
        - Type 4: Olive skin, burns minimally, tans easily
        - Type 5: Brown skin, rarely burns, tans darkly
        - Type 6: Dark brown/black skin, never burns, always tans
        """)
        
        # Medical History Section
        st.markdown("### Medical History")
        st.markdown("""
        Medical history provides important context for risk assessment and diagnosis.
        """)
        
        # Input fields for medical history
        st.session_state.smoking = st.radio("Smoking Status", ["Never", "Former", "Current"], index=["Never", "Former", "Current"].index(st.session_state.smoking))
        st.info("Smoking history can affect skin health and healing.")
        
        st.session_state.cancer_history = st.radio("Family History of Cancer", ["Yes", "No"], index=0 if st.session_state.cancer_history == "Yes" else 1)
        st.info("Family history of any type of cancer can indicate genetic predisposition.")
        
        st.session_state.skin_cancer_history = st.radio("Personal History of Skin Cancer", ["Yes", "No"], index=0 if st.session_state.skin_cancer_history == "Yes" else 1)
        st.info("Previous skin cancer diagnosis significantly increases risk of new occurrences.")
        
        # Lesion Information Section
        st.markdown("### Lesion Information")
        st.markdown("""
        Detailed information about the skin lesion helps in accurate assessment.
        All measurements should be taken with a ruler or measuring tape in millimeters (mm).
        """)
        
        # Input fields for lesion information
        # Use 'location' for UI state but map to 'region' for model
        locations = ["ARM", "NECK", "FACE", "HAND", "FOREARM", "CHEST", "OTHER"]
        st.session_state.location = st.selectbox("Lesion Location", locations, index=locations.index(st.session_state.location))
        st.info("Location of the lesion on the body. Different locations may indicate different types of skin cancer.")
        
        st.markdown("#### Lesion Dimensions")
        st.markdown("""
        Measure the lesion in two perpendicular directions using a ruler or measuring tape.
        Record measurements in millimeters (mm) to the nearest 0.1 mm.
        """)
        
        st.session_state.diameter_1 = st.slider("Primary Diameter (mm)", 1.0, 20.0, st.session_state.diameter_1, step=0.1)
        st.info("Longest diameter of the lesion in millimeters (mm)")
        
        st.session_state.diameter_2 = st.slider("Secondary Diameter (mm)", 1.0, 20.0, st.session_state.diameter_2, step=0.1)
        st.info("Diameter perpendicular to the primary diameter in millimeters (mm)")
        
        # Symptoms Section
        st.markdown("### Symptoms")
        st.markdown("""
        Document any symptoms associated with the lesion. These can be important indicators of malignancy.
        Timeframes for symptoms should be noted in weeks or months.
        """)
        
        # Input fields for symptoms - use UI-friendly names
        st.session_state.itching = st.radio("Itching", ["Yes", "No"], index=0 if st.session_state.itching == "Yes" else 1)
        st.info("Persistent itching can be a sign of certain types of skin cancer. Note duration in weeks/months.")
        
        st.session_state.growing = st.radio("Is the lesion growing?", ["Yes", "No"], index=0 if st.session_state.growing == "Yes" else 1)
        st.info("Rapid growth or change in size is a concerning feature. Note growth rate in mm per month if known.")
        
        st.session_state.pain = st.radio("Pain or Tenderness", ["Yes", "No"], index=0 if st.session_state.pain == "Yes" else 1) # Renamed label slightly
        st.info("Pain or tenderness in the lesion should be noted. Rate pain on a scale of 1-10 if present.")
        
        st.session_state.bleeding = st.radio("Bleeding", ["Yes", "No"], index=0 if st.session_state.bleeding == "Yes" else 1)
        st.info("Spontaneous bleeding or easy bleeding with minor trauma. Note frequency (times per week/month).")
        
        # Note: Elevation might not be used by the model based on warnings, kept in UI state for now
        st.session_state.elevation = st.radio("Is the lesion elevated?", ["Yes", "No"], index=0 if st.session_state.elevation == "Yes" else 1)
        st.info("Note if the lesion is raised above the skin surface. Measure height in millimeters if possible.")
        
        # Environmental Factors Section
        st.markdown("### Environmental Factors")
        st.markdown("""
        Environmental exposures can contribute to skin cancer risk.
        Duration of exposure should be noted in years where applicable.
        """)
        
        # Input fields for environmental factors - use UI-friendly names
        st.session_state.pesticide_exposure = st.radio("Pesticide Exposure", ["Yes", "No"], index=0 if st.session_state.pesticide_exposure == "Yes" else 1)
        st.info("History of occupational or significant environmental pesticide exposure. Note duration in years.")
        
        st.session_state.piped_water = st.radio("Access to Piped Water", ["Yes", "No"], index=0 if st.session_state.piped_water == "Yes" else 1)
        st.info("Access to clean water can affect overall skin health.")
        
        st.session_state.sewage_system = st.radio("Access to Sewage System", ["Yes", "No"], index=0 if st.session_state.sewage_system == "Yes" else 1)
        st.info("Sanitation conditions can impact skin health.")
        
        # Submit button for analysis
        submit_button = st.button("Submit for Analysis", disabled=not st.session_state.model_loaded)
        
        if not st.session_state.model_loaded:
            st.warning("Model not loaded. Please train the model first by running `skin_cancer_model_clean.py`.")
            
        if submit_button:
            with st.spinner("Analyzing data..."):
                try:
                    # Prepare input data for prediction
                    input_data = prepare_prediction_input()
                    
                    # Make prediction
                    st.session_state.clinical_prediction_results = predict(input_data)
                    
                    # Check if there was an error
                    if "error" in st.session_state.clinical_prediction_results and st.session_state.clinical_prediction_results["error"]:
                        st.error(f"Error making prediction: {st.session_state.clinical_prediction_results['error']}")
                        st.session_state.clinical_prediction_made = False
                    else:
                        st.session_state.clinical_prediction_made = True
                        
                        # Get prediction details
                        prediction_type = "Clinical Features"
                        prediction_result = st.session_state.clinical_prediction_results.get("prediction", "Unknown")
                        confidence = st.session_state.clinical_prediction_results.get("confidence", 0)
                        
                        # Ensure confidence is a float
                        try:
                            confidence = float(confidence)
                        except (ValueError, TypeError):
                            confidence = 0.0
                        
                        # Determine recommendation
                        if prediction_result in ["MEL", "SCC", "BCC"]:
                            recommendation = "Urgent dermatologist consultation recommended"
                        elif prediction_result in ["AKIEC"]:
                            recommendation = "Dermatologist consultation recommended"
                        else:
                            recommendation = "Regular follow-up recommended"
                        
                        # Save to database (Ensure only expected arguments are passed)
                        pred_result = save_prediction(
                            st.session_state.user['id'],
                            prediction_type,
                            prediction_result,
                            confidence,
                            recommendation,
                            input_data  # Clinical features as dict
                        )
                        
                        if pred_result['success']:
                            st.session_state.current_prediction_id = pred_result['prediction_id']
                        else:
                            logger.error(f"Error saving prediction: {pred_result['error']}")
                    
                    # Log prediction results
                    logger.info(f"Prediction made: {st.session_state.clinical_prediction_results}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    logger.error(f"Error making prediction: {str(e)}")
                    st.session_state.clinical_prediction_made = False

# Show prediction and appointment booking if prediction is made
if st.session_state.image_prediction_made or st.session_state.clinical_prediction_made:
    st.markdown("---")
    st.header("Analysis Results")
    
    # Initialize variables
    prediction = "Unknown"
    confidence = 0
    recommendation = "Regular follow-up recommended"
    sorted_probs = []
    current_results = None
    show_appointment_booking = False
    
    # Determine which prediction to use based on input method
    if input_method == "Image Upload" and st.session_state.image_prediction_made and st.session_state.image_prediction_results:
        current_results = st.session_state.image_prediction_results
        prediction = current_results.get("prediction", "Unknown")
        confidence = current_results.get("confidence", 0)
        sorted_probs = sorted(current_results.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)
        show_appointment_booking = True
        
        # Success message with prediction results
        st.success("Image Analysis Complete!")
        
        # Display main prediction result for image analysis
        confidence_display = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else "N/A"
        st.markdown(f"""
        ### Preliminary Assessment
        - **Diagnosis**: {prediction}
        - **Confidence**: {confidence_display}
        """)
        
        # Display recommendation based on prediction
        if "melanoma" in prediction.lower():
            recommendation = "Urgent dermatologist consultation recommended"
            st.error(f"**Recommendation**: {recommendation}")
        elif "basal cell carcinoma" in prediction.lower() or "actinic keratoses" in prediction.lower():
            recommendation = "Dermatologist consultation recommended"
            st.warning(f"**Recommendation**: {recommendation}")
        else:
            recommendation = "Regular follow-up recommended"
            st.info(f"**Recommendation**: {recommendation}")
        
        # Display probabilities for all classes if available
        if sorted_probs:
            st.markdown("### Detailed Probabilities")
            
            # Create data for bar chart
            chart_data = pd.DataFrame(
                [{"Class": k, "Probability (%)": v} for k, v in sorted_probs]
            )
            
            # Display bar chart
            st.bar_chart(chart_data.set_index("Class"))
            
            # Display table of probabilities
            st.markdown("#### All Class Probabilities")
            probability_table = []
            
            for class_label, prob in sorted_probs:
                display_name = class_label
                
                probability_table.append({
                    "Class": display_name,
                    "Probability (%)": f"{prob:.2f}%"
                })
            
            st.table(probability_table)
    
    elif input_method == "Clinical Features" and st.session_state.clinical_prediction_made and st.session_state.clinical_prediction_results:
        current_results = st.session_state.clinical_prediction_results
        prediction = current_results.get("prediction", "Unknown")
        confidence = current_results.get("confidence", 0)
        sorted_probs = sorted(current_results.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)
        show_appointment_booking = True
        
        # Success message with prediction results
        st.success("Clinical Analysis Complete!")
        
        # Display main prediction result for clinical features
        confidence_display = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else "N/A"
        st.markdown(f"""
        ### Preliminary Assessment
        - **Diagnosis**: {prediction}
        - **Confidence**: {confidence_display}
        """)
        
        # Display recommendation based on prediction
        if prediction in ["MEL", "SCC", "BCC"]:
            recommendation = "Urgent dermatologist consultation recommended"
            st.error(f"**Recommendation**: {recommendation}")
        elif prediction in ["AKIEC"]:
            recommendation = "Dermatologist consultation recommended"
            st.warning(f"**Recommendation**: {recommendation}")
        else:
            recommendation = "Regular follow-up recommended"
            st.info(f"**Recommendation**: {recommendation}")
        
        # Display probabilities for all classes if available
        if sorted_probs:
            st.markdown("### Detailed Probabilities")
            
            # Create data for bar chart
            chart_data = pd.DataFrame(
                [{"Class": k, "Probability (%)": v} for k, v in sorted_probs]
            )
            
            # Display bar chart
            st.bar_chart(chart_data.set_index("Class"))
            
            # Display table of probabilities
            st.markdown("#### All Class Probabilities")
            probability_table = []
            
            # Create a dictionary mapping class codes to full names
            class_names = {
                "MEL": "Melanoma",
                "NV": "Melanocytic Nevus",
                "BCC": "Basal Cell Carcinoma",
                "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma",
                "BKL": "Benign Keratosis",
                "DF": "Dermatofibroma",
                "VASC": "Vascular Lesion"
            }
            
            for class_label, prob in sorted_probs:
                display_name = class_names.get(class_label, class_label)
                
                probability_table.append({
                    "Class": display_name,
                    "Probability (%)": f"{prob:.2f}%"
                })
            
            st.table(probability_table)
    else:
        if input_method == "Image Upload":
            st.warning("Please complete the image analysis before viewing results or booking an appointment.")
        else:
            st.warning("Please complete the clinical features analysis before viewing results or booking an appointment.")

    # Appointment booking section
    st.markdown("---")
    st.header("Book an Appointment")
    
    # Only show appointment booking if we have valid prediction results
    if show_appointment_booking:
        # Decide default urgency based on recommendation
        urgency = "Normal"
        if "urgent" in recommendation.lower():
            urgency = "Urgent"
        elif "recommended" in recommendation.lower():
            urgency = "Soon"
        
        st.markdown(f"""
        Based on the preliminary assessment, we recommend scheduling a {'**URGENT**' if urgency == 'Urgent' else ''} appointment 
        with a dermatologist for further evaluation and confirmation of the diagnosis.
        """)
        
        # Appointment form
        with st.form("appointment_form"):
            st.markdown("### Appointment Details")
            
            # Today's date
            today = datetime.now().date()
            
            # Default appointment date based on urgency
            default_date = today
            if urgency == "Urgent":
                default_date = today + timedelta(days=1)
            elif urgency == "Soon":
                default_date = today + timedelta(days=3)
            else:
                default_date = today + timedelta(days=7)
            
            # Appointment date picker
            appointment_date = st.date_input(
                "Preferred Date",
                value=default_date,
                min_value=today,
                max_value=today + timedelta(days=30)
            )
            
            # Time slots based on urgency
            time_slots = []
            if urgency == "Urgent":
                time_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM", "5:00 PM"]
            else:
                time_slots = ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"]
            
            # Time slot selection
            time_slot = st.selectbox("Preferred Time", time_slots)
            
            # Doctor selection
            doctors = [
                "Dr. Ahmad Rashid (Dermatologist)",
                "Dr. Fatima Khan (Skin Cancer Specialist)",
                "Dr. Muhammad Ali (Dermatopathologist)",
                "Dr. Aisha Malik (Dermatologist)"
            ]
            doctor = st.selectbox("Preferred Doctor", doctors)
            
            # Additional notes
            notes = st.text_area("Additional Notes", "")
            
            # Submit button
            submitted = st.form_submit_button("Book Appointment")
            
            if submitted:
                # Save appointment to database
                if st.session_state.current_prediction_id:
                    # Format appointment date for database
                    appointment_date_str = appointment_date.strftime('%Y-%m-%d')
                    
                    # Format confidence safely 
                    confidence_display = f"{confidence:.1f}%"
                    
                    # Save appointment
                    appt_result = save_appointment(
                        st.session_state.user['id'],
                        st.session_state.current_prediction_id,
                        appointment_date_str,
                        time_slot,
                        doctor,
                        notes
                    )
                    
                    if appt_result['success']:
                        st.success(f"""
                        Appointment Booked Successfully!
                        
                        - **Patient**: {st.session_state.patient_name}
                        - **Date**: {appointment_date.strftime('%B %d, %Y')}
                        - **Time**: {time_slot}
                        - **Doctor**: {doctor}
                        
                        **Prediction Results**:
                        - **Diagnosis**: {prediction}
                        - **Confidence**: {confidence_display}
                        - **Recommendation**: {recommendation}
                        
                        A confirmation email has been sent to {st.session_state.patient_email}.
                        Please arrive 15 minutes before your scheduled appointment time.
                        """)
                    else:
                        st.error(f"Error booking appointment: {appt_result['error']}")
                else:
                    st.error("Error: No prediction data available for this appointment")
        
        # Add download button for results if we have valid prediction results
        if show_appointment_booking and current_results and sorted_probs:
            st.markdown("---")
            st.subheader("Download Results")
            
            # Determine which class names dictionary to use
            if input_method == "Image Upload":
                # For image analysis
                class_display_names = {k: k for k, v in current_results.get("probabilities", {}).items()}
            else:
                # For clinical features
                class_display_names = {
                    "MEL": "Melanoma",
                    "NV": "Melanocytic Nevus",
                    "BCC": "Basal Cell Carcinoma",
                    "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma",
                    "BKL": "Benign Keratosis",
                    "DF": "Dermatofibroma",
                    "VASC": "Vascular Lesion"
                }
            
            result_summary = f"""
            SKIN CANCER CLASSIFICATION REPORT
            ================================
            
            PATIENT INFORMATION:
            -------------------
            Name: {st.session_state.patient_name}
            Email: {st.session_state.patient_email}
            Phone: {st.session_state.patient_phone}
            Address: {st.session_state.patient_address}
            
            REPORT DATE: {datetime.now().strftime('%B %d, %Y')}
            
            ASSESSMENT METHOD: {"Image Analysis" if input_method == "Image Upload" else "Clinical Features Analysis"}
            
            """
            
            # Add clinical features if using that method
            if input_method == "Clinical Features":
                result_summary += f"""
            CLINICAL FEATURES:
            -----------------
            Age: {st.session_state.age} years
            Gender: {st.session_state.gender}
            Fitzpatrick Skin Type: {st.session_state.fitzpatrick}
            Lesion Location: {st.session_state.location}
            Lesion Dimensions: {st.session_state.diameter_1}mm x {st.session_state.diameter_2}mm
            
            Symptoms:
            - Itching: {st.session_state.itching}
            - Growing: {st.session_state.growing}
            - Pain: {st.session_state.pain}
            - Bleeding: {st.session_state.bleeding}
            - Elevation: {st.session_state.elevation}
            
            Medical History:
            - Smoking Status: {st.session_state.smoking}
            - Family History of Cancer: {st.session_state.cancer_history}
            - Personal History of Skin Cancer: {st.session_state.skin_cancer_history}
            
            Environmental Factors:
            - Pesticide Exposure: {st.session_state.pesticide_exposure}
            - Piped Water: {st.session_state.piped_water}
            - Sewage System: {st.session_state.sewage_system}
            """
            
            result_summary += f"""
            ASSESSMENT RESULTS:
            -----------------
            Primary Diagnosis: {prediction}
            Confidence Level: {confidence:.1f}%
            
            Detailed Class Probabilities:
            """
            
            # Add probabilities for each class
            for class_label, prob in sorted_probs:
                display_name = class_display_names.get(class_label, class_label)
                result_summary += f"- {display_name}: {prob:.2f}%\n"
            
            result_summary += f"""
            
            RECOMMENDATION:
            --------------
            {recommendation}
            
            APPOINTMENT DETAILS:
            ------------------
            Date: {appointment_date.strftime('%B %d, %Y')}
            Time: {time_slot}
            Doctor: {doctor}
            
            IMPORTANT NOTES:
            ---------------
            1. This is a preliminary assessment and not a clinical diagnosis.
            2. The results are based on machine learning analysis of provided information.
            3. Please consult with a healthcare professional for proper diagnosis and treatment.
            4. Bring this report to your appointment for reference.
            5. Arrive 15 minutes before your scheduled appointment time.
            
            CONFIDENTIALITY NOTICE:
            ----------------------
            This report contains confidential medical information. Unauthorized disclosure is prohibited.
            """
            
            st.download_button(
                label="Download Report as TXT",
                data=result_summary,
                file_name=f"skin_cancer_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
            
            # Add link to educational resources
            st.markdown("---")
            st.subheader("Educational Resources")
            st.markdown("""
            Learn more about skin cancer types, prevention, and treatment:
            
            - [American Academy of Dermatology](https://www.aad.org/public/diseases/skin-cancer)
            - [Skin Cancer Foundation](https://www.skincancer.org/)
            - [National Cancer Institute](https://www.cancer.gov/types/skin)
            - [World Health Organization - Skin Cancer](https://www.who.int/news-room/fact-sheets/detail/ultraviolet-radiation)
            """)
            
            # What to expect at appointment
            st.markdown("---")
            st.subheader("What to Expect at Your Appointment")
            st.markdown("""
            During your dermatology appointment, you can expect:
            
            1. **Review of your medical history**: The doctor will ask about your personal and family health history.
            2. **Full skin examination**: A thorough check of your skin, including the lesion of concern.
            3. **Dermoscopy**: The use of a special magnifying tool to examine the lesion more closely.
            4. **Biopsy**: If necessary, a small sample of tissue may be taken for laboratory analysis.
            5. **Discussion of findings**: The doctor will explain their findings and recommend next steps.
            6. **Treatment plan**: If needed, a treatment plan will be developed based on the diagnosis.
            
            Please bring a list of your current medications and any previous dermatology records to your appointment.
            """)

# Disclaimer and footer
st.markdown("---")
st.caption("""
**Disclaimer**: This application is intended for educational and informational purposes only and does not constitute medical advice. 
The predictions made by this system are preliminary assessments and should not be considered as a clinical diagnosis. 
Always consult with a qualified healthcare professional for proper diagnosis, advice, and treatment.
""")
st.caption("© 2023 Skin Cancer Classification System - All Rights Reserved") 