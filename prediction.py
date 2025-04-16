"""
Prediction module for skin cancer classification web application.
This module loads the trained model and preprocessing objects and provides
functions to preprocess input data and make predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to find the model directory
def find_model_dir():
    """
    Look for the model directory in various possible locations
    """
    # List of possible relative paths to check
    possible_paths = [
        "../model",                   # Standard relative path from web_app
        "./model",                    # If app is run from root
        "../../model",                # Another possible location
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")  # Absolute path from module location
    ]
    
    # Check each path
    for path in possible_paths:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "skin_cancer_model.joblib")):
            logger.info(f"Found model directory at: {path}")
            return path
    
    # If we can't find the model directory, log an error and return None
    logger.error("Could not find model directory in any known location")
    return None

# Get model directory path
MODEL_DIR = find_model_dir()

# Function to load model and preprocessing objects
def load_model_and_preprocessors():
    """
    Load the trained model and all preprocessing objects.
    Returns a dictionary containing the model and preprocessors.
    """
    if MODEL_DIR is None:
        logger.error("Model directory not found. Please run skin_cancer_model_clean.py first.")
        return None
    
    try:
        # Paths to model and preprocessing objects
        model_path = os.path.join(MODEL_DIR, "skin_cancer_model.joblib")
        info_path = os.path.join(MODEL_DIR, "preprocessing_info.joblib")
        num_imputer_path = os.path.join(MODEL_DIR, "numerical_imputer.joblib")
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        cat_encoder_path = os.path.join(MODEL_DIR, "categorical_encoder.joblib")
        label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
        
        # Check if main model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        # Log which files we're loading
        logger.info(f"Loading model from: {model_path}")
        
        # Load model and preprocessing objects
        model = joblib.load(model_path)
        
        # Check if preprocessing info exists
        if not os.path.exists(info_path):
            logger.error(f"Preprocessing info file not found at: {info_path}")
            return None
            
        preprocessing_info = joblib.load(info_path)
        
        # Check if label encoder exists
        if not os.path.exists(label_encoder_path):
            logger.error(f"Label encoder file not found at: {label_encoder_path}")
            return None
            
        label_encoder = joblib.load(label_encoder_path)
        
        # Load optional preprocessing objects if they exist
        num_imputer = None
        scaler = None
        cat_encoder = None
        
        if os.path.exists(num_imputer_path):
            logger.info(f"Loading numerical imputer from: {num_imputer_path}")
            num_imputer = joblib.load(num_imputer_path)
        else:
            logger.warning(f"Numerical imputer not found at: {num_imputer_path}")
        
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from: {scaler_path}")
            scaler = joblib.load(scaler_path)
        else:
            logger.warning(f"Scaler not found at: {scaler_path}")
            
        if os.path.exists(cat_encoder_path):
            logger.info(f"Loading categorical encoder from: {cat_encoder_path}")
            cat_encoder = joblib.load(cat_encoder_path)
        else:
            logger.warning(f"Categorical encoder not found at: {cat_encoder_path}")
        
        logger.info("Successfully loaded model and preprocessing objects")
        
        # Return all loaded objects
        return {
            "model": model,
            "preprocessing_info": preprocessing_info,
            "num_imputer": num_imputer,
            "scaler": scaler,
            "cat_encoder": cat_encoder,
            "label_encoder": label_encoder
        }
    
    except Exception as e:
        logger.error(f"Error loading model and preprocessing objects: {e}")
        return None

# Function to preprocess input data
def preprocess_input(input_data, model_objects):
    """
    Preprocess input data using the trained preprocessing objects.
    
    Args:
        input_data (dict): Dictionary containing input data from the web app
        model_objects (dict): Dictionary containing model and preprocessing objects
    
    Returns:
        numpy.ndarray: Preprocessed input data ready for prediction
    """
    try:
        # Extract preprocessing info and objects
        info = model_objects["preprocessing_info"]
        num_imputer = model_objects["num_imputer"]
        scaler = model_objects["scaler"]
        cat_encoder = model_objects["cat_encoder"]
        
        # Extract feature lists
        numerical_features = info["numerical_features"]
        all_cat_features = info["all_cat_features"]
        
        # Log what we're preprocessing
        logger.info(f"Preprocessing input data with {len(numerical_features)} numerical features and {len(all_cat_features)} categorical features")
        
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Check that all required columns are present
        missing_num_cols = [col for col in numerical_features if col not in input_df.columns]
        missing_cat_cols = [col for col in all_cat_features if col not in input_df.columns]
        
        if missing_num_cols:
            logger.warning(f"Missing numerical columns in input data: {missing_num_cols}")
            # Add missing columns with NaN values
            for col in missing_num_cols:
                input_df[col] = np.nan
                
        if missing_cat_cols:
            logger.warning(f"Missing categorical columns in input data: {missing_cat_cols}")
            # Add missing columns with NaN values
            for col in missing_cat_cols:
                input_df[col] = np.nan
        
        # Handle numerical features
        if numerical_features and num_imputer and scaler:
            # Impute missing values
            input_df[numerical_features] = num_imputer.transform(input_df[numerical_features])
            # Scale numerical features
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        else:
            if not num_imputer:
                logger.warning("Numerical imputer not available for preprocessing")
            if not scaler:
                logger.warning("Scaler not available for preprocessing")
        
        # Handle categorical features
        encoded_cats = None
        if all_cat_features and cat_encoder:
            # Fill missing values with 'missing'
            for col in all_cat_features:
                if col in input_df.columns:
                    input_df[col] = input_df[col].fillna('missing').astype(str)
            
            # One-hot encode categorical features
            encoded_cats = cat_encoder.transform(input_df[all_cat_features])
        else:
            if not cat_encoder:
                logger.warning("Categorical encoder not available for preprocessing")
        
        # Combine numerical and categorical features
        if numerical_features:
            if encoded_cats is not None:
                final_features = np.hstack([input_df[numerical_features].values, encoded_cats])
                logger.info(f"Final feature matrix shape: {final_features.shape}")
            else:
                final_features = input_df[numerical_features].values
                logger.info(f"Using only numerical features with shape: {final_features.shape}")
        else:
            if encoded_cats is not None:
                final_features = encoded_cats
                logger.info(f"Using only categorical features with shape: {final_features.shape}")
            else:
                # No features to process
                logger.error("No features available for preprocessing")
                return None
        
        return final_features
    
    except Exception as e:
        logger.error(f"Error preprocessing input data: {e}")
        logger.exception("Detailed exception info:")
        return None

# Function to make prediction
def predict(input_data):
    """
    Make prediction using the trained model.
    
    Args:
        input_data (dict): Dictionary containing input data from the web app
    
    Returns:
        dict: Dictionary containing prediction results
    """
    try:
        # Load model and preprocessors
        model_objects = load_model_and_preprocessors()
        if not model_objects:
            logger.error("Could not load model and preprocessing objects")
            return {
                "error": "Failed to load model and preprocessing objects. Please ensure the model is trained and available.",
                "prediction": "Unknown",
                "confidence": 0,
                "probabilities": {}
            }
        
        # Preprocess input data
        preprocessed_data = preprocess_input(input_data, model_objects)
        if preprocessed_data is None:
            logger.error("Failed to preprocess input data")
            return {
                "error": "Failed to preprocess input data. Please check that all required fields are provided.",
                "prediction": "Unknown",
                "confidence": 0,
                "probabilities": {}
            }
        
        # Make prediction
        model = model_objects["model"]
        label_encoder = model_objects["label_encoder"]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(preprocessed_data)[0]
        
        # Get class names
        class_names = label_encoder.classes_
        
        # Find highest probability and corresponding class
        max_prob_index = np.argmax(probabilities)
        prediction = class_names[max_prob_index]
        confidence = probabilities[max_prob_index] * 100
        
        # Create dictionary of class probabilities
        class_probabilities = {
            class_name: float(prob) * 100 
            for class_name, prob in zip(class_names, probabilities)
        }
        
        logger.info(f"Prediction made: {prediction} with confidence {confidence:.2f}%")
        
        # Return prediction results
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": class_probabilities
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        logger.exception("Detailed exception info:")
        return {
            "error": str(e),
            "prediction": "Unknown",
            "confidence": 0,
            "probabilities": {}
        }
