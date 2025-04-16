import firebase_admin
from firebase_admin import credentials, firestore, auth
import hashlib
import json
import uuid
from datetime import datetime, timedelta
import streamlit as st
import os
import logging

# Initialize Firebase
def init_firebase():
    """Initialize Firebase Admin SDK connection"""
    if not firebase_admin._apps:  # Check if not already initialized
        # For local development, use service account
        if os.path.exists('firebase-key.json'):
            cred = credentials.Certificate('firebase-key.json')
            firebase_admin.initialize_app(cred)
        # For Streamlit Cloud deployment, use secrets
        elif 'firebase' in st.secrets:
            # Convert the dictionary to a proper service account JSON
            key_dict = st.secrets["firebase"]
            # Initialize the app with the service account credentials
            cred = credentials.Certificate(key_dict)
            firebase_admin.initialize_app(cred)
        else:
            raise ValueError("Firebase credentials not found")
    
    return firestore.client()

# User management functions
def hash_password(password):
    """Hash a password for storing"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against a provided password"""
    return stored_password == hash_password(provided_password)

def create_user(username, password, email, full_name, phone=None, address=None):
    """Create a new user in Firebase"""
    try:
        db = init_firebase()
        # Check if username exists
        users_ref = db.collection('users')
        username_query = users_ref.where('username', '==', username).get()
        if len(list(username_query)) > 0:
            return {'success': False, 'error': 'Username already exists'}
        
        # Check if email exists
        email_query = users_ref.where('email', '==', email).get()
        if len(list(email_query)) > 0:
            return {'success': False, 'error': 'Email already exists'}
            
        # Create user
        hashed_password = hash_password(password)
        user_data = {
            'username': username,
            'password_hash': hashed_password,
            'email': email,
            'full_name': full_name,
            'phone': phone,
            'address': address,
            'role': 'user',  # Default role
            'created_at': firestore.SERVER_TIMESTAMP
        }
        
        # Add user to Firestore
        user_ref = users_ref.document()
        user_ref.set(user_data)
        
        return {'success': True, 'user_id': user_ref.id}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_by_username(username):
    """Get user by username from Firebase"""
    try:
        db = init_firebase()
        users_ref = db.collection('users')
        query = users_ref.where('username', '==', username).limit(1).get()
        
        user_docs = list(query)
        if not user_docs:
            return None
            
        user_data = user_docs[0].to_dict()
        user_data['id'] = user_docs[0].id  # Add the document ID as 'id'
        return user_data
    except Exception as e:
        # print(f"Error getting user: {e}") # Removed debug print
        logger.error(f"Error getting user by username '{username}': {e}") # Added logger instead
        return None

def login_user(username, password):
    """Login a user by username and password using Firebase"""
    user = get_user_by_username(username)
    if user and verify_password(user['password_hash'], password):
        # Remove password_hash from user data before returning
        user_data = {k: user[k] for k in user.keys() if k != 'password_hash'}
        return {'success': True, 'user': user_data}
    return {'success': False, 'error': 'Invalid username or password'}

def update_user_profile(user_id, full_name=None, email=None, phone=None, address=None):
    """Update user profile information in Firebase"""
    try:
        db = init_firebase()
        user_ref = db.collection('users').document(user_id)
        
        # Check if email exists and is different
        if email:
            email_query = db.collection('users').where('email', '==', email).get()
            for doc in email_query:
                if doc.id != user_id:
                    return {'success': False, 'error': 'Email already exists'}
        
        # Build update data
        update_data = {}
        if full_name:
            update_data['full_name'] = full_name
        if email:
            update_data['email'] = email
        if phone:
            update_data['phone'] = phone
        if address:
            update_data['address'] = address
        
        if not update_data:
            return {'success': False, 'error': 'No updates provided'}
        
        # Update timestamp
        update_data['updated_at'] = firestore.SERVER_TIMESTAMP
        
        # Update user document
        user_ref.update(update_data)
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Session management
def create_session(user_id, expires_days=30):
    """Create a session for a user in Firebase"""
    try:
        db = init_firebase()
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(days=expires_days)
        
        session_data = {
            'user_id': user_id,
            'created_at': firestore.SERVER_TIMESTAMP,
            'expires_at': expires_at,
            'is_active': True
        }
        
        db.collection('sessions').document(session_id).set(session_data)
        return {'success': True, 'session_id': session_id}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_session(session_id):
    """Get session details from Firebase"""
    try:
        db = init_firebase()
        session_ref = db.collection('sessions').document(session_id)
        session = session_ref.get()
        
        if not session.exists:
            return {'success': False, 'error': 'Session not found'}
        
        session_data = session.to_dict()
        
        # Check if session is expired
        expires_at = session_data.get('expires_at')
        if expires_at and expires_at.timestamp() < datetime.now().timestamp():
            # Delete expired session
            session_ref.delete()
            return {'success': False, 'error': 'Session expired'}
        
        # Get user data
        user_id = session_data.get('user_id')
        user_ref = db.collection('users').document(user_id)
        user = user_ref.get()
        
        if not user.exists:
            return {'success': False, 'error': 'User not found'}
        
        user_data = user.to_dict()
        user_data['id'] = user.id
        
        # Remove password hash
        if 'password_hash' in user_data:
            del user_data['password_hash']
        
        return {'success': True, 'user': user_data}
    except Exception as e:
        # print(f"Error getting session: {e}") # Removed debug print
        logger.error(f"Error getting session '{session_id}': {e}") # Added logger instead
        return {'success': False, 'error': str(e)}

def delete_session(session_id):
    """Delete a session from Firebase"""
    try:
        db = init_firebase()
        db.collection('sessions').document(session_id).delete()
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Prediction and appointment functions
def save_prediction(user_id, prediction_type, prediction_result, confidence, recommendation, 
                    clinical_features=None):
    """Save a prediction to Firebase (without image path)"""
    try:
        db = init_firebase()
        
        # Prepare clinical features
        # Firestore handles dict directly
        if clinical_features and not isinstance(clinical_features, dict):
             # Attempt to handle if it's accidentally passed as JSON string (though it shouldn't be)
             try:
                 clinical_features = json.loads(clinical_features)
             except Exception:
                 clinical_features = str(clinical_features) # Store as string if parsing fails
            
        prediction_data = {
            'user_id': user_id,
            'prediction_type': prediction_type,
            'prediction_result': prediction_result,
            'confidence': confidence,
            'recommendation': recommendation,
            'clinical_features': clinical_features, # Can be None or dict
            # 'image_path': image_path, # REMOVED
            'created_at': firestore.SERVER_TIMESTAMP
        }
        
        # Remove None values if desired, or keep them
        # prediction_data = {k: v for k, v in prediction_data.items() if v is not None}

        prediction_ref = db.collection('predictions').document()
        prediction_ref.set(prediction_data)
        
        return {'success': True, 'prediction_id': prediction_ref.id}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_predictions(user_id):
    """Get prediction history for a user from Firebase"""
    try:
        db = init_firebase()
        predictions_ref = db.collection('predictions')
        query = predictions_ref.where('user_id', '==', user_id).order_by('created_at', direction=firestore.Query.DESCENDING)
        
        predictions = []
        for doc in query.stream():
            prediction_data = doc.to_dict()
            prediction_data['id'] = doc.id
            predictions.append(prediction_data)
            
        return predictions
    except Exception as e:
        # print(f"Error getting predictions: {e}") # Removed debug print
        logger.error(f"Error getting predictions for user '{user_id}': {e}") # Added logger instead
        return []

def get_all_predictions(limit=100):
    """Get all predictions from Firebase (admin function)"""
    try:
        db = init_firebase()
        predictions_ref = db.collection('predictions')
        query = predictions_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)
        
        predictions = []
        for doc in query.stream():
            prediction_data = doc.to_dict()
            prediction_data['id'] = doc.id
            
            # Get user info
            user_id = prediction_data.get('user_id')
            user_ref = db.collection('users').document(user_id)
            user = user_ref.get()
            
            if user.exists:
                user_data = user.to_dict()
                prediction_data['username'] = user_data.get('username')
                prediction_data['full_name'] = user_data.get('full_name')
                prediction_data['email'] = user_data.get('email')
            
            predictions.append(prediction_data)
            
        return predictions
    except Exception as e:
        # print(f"Error getting all predictions: {e}") # Removed debug print
        logger.error(f"Error getting all predictions: {e}") # Added logger instead
        return []

def save_appointment(user_id, prediction_id, appointment_date, appointment_time, doctor, notes=None):
    """Save an appointment to Firebase"""
    try:
        db = init_firebase()
        
        appointment_data = {
            'user_id': user_id,
            'prediction_id': prediction_id,
            'appointment_date': appointment_date,
            'appointment_time': appointment_time,
            'doctor': doctor,
            'notes': notes,
            'created_at': firestore.SERVER_TIMESTAMP,
            'status': 'scheduled'  # Default status
        }
        
        appointment_ref = db.collection('appointments').document()
        appointment_ref.set(appointment_data)
        
        return {'success': True, 'appointment_id': appointment_ref.id}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_user_appointments(user_id):
    """Get appointments for a user from Firebase"""
    try:
        db = init_firebase()
        appointments_ref = db.collection('appointments')
        query = appointments_ref.where('user_id', '==', user_id).order_by('appointment_date')
        
        appointments = []
        for doc in query.stream():
            appointment_data = doc.to_dict()
            appointment_data['id'] = doc.id
            
            # Get prediction info
            prediction_id = appointment_data.get('prediction_id')
            if prediction_id:
                prediction_ref = db.collection('predictions').document(prediction_id)
                prediction = prediction_ref.get()
                
                if prediction.exists:
                    prediction_data = prediction.to_dict()
                    appointment_data['prediction_result'] = prediction_data.get('prediction_result')
                    appointment_data['confidence'] = prediction_data.get('confidence')
            
            appointments.append(appointment_data)
            
        return appointments
    except Exception as e:
        # print(f"Error getting appointments: {e}") # Removed debug print
        logger.error(f"Error getting appointments for user '{user_id}': {e}") # Added logger instead
        return []

def get_all_appointments():
    """Get all appointments from Firebase (admin function)"""
    try:
        db = init_firebase()
        appointments_ref = db.collection('appointments')
        query = appointments_ref.order_by('appointment_date')
        
        appointments = []
        for doc in query.stream():
            appointment_data = doc.to_dict()
            appointment_data['id'] = doc.id
            
            # Get user info
            user_id = appointment_data.get('user_id')
            user_ref = db.collection('users').document(user_id)
            user = user_ref.get()
            
            if user.exists:
                user_data = user.to_dict()
                appointment_data['username'] = user_data.get('username')
                appointment_data['full_name'] = user_data.get('full_name')
                appointment_data['email'] = user_data.get('email')
            
            # Get prediction info
            prediction_id = appointment_data.get('prediction_id')
            if prediction_id:
                prediction_ref = db.collection('predictions').document(prediction_id)
                prediction = prediction_ref.get()
                
                if prediction.exists:
                    prediction_data = prediction.to_dict()
                    appointment_data['prediction_result'] = prediction_data.get('prediction_result')
                    appointment_data['confidence'] = prediction_data.get('confidence')
            
            appointments.append(appointment_data)
            
        return appointments
    except Exception as e:
        # print(f"Error getting all appointments: {e}") # Removed debug print
        logger.error(f"Error getting all appointments: {e}") # Added logger instead
        return [] 