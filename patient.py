import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.firebase_utils import get_user_predictions, get_user_appointments
from auth import require_login, profile_management

def patient_dashboard():
    """Display patient dashboard"""
    # Make sure the user is logged in
    user = require_login()
    
    st.title(f"Welcome, {user.get('full_name', 'User')}") # Use .get for safety
    st.markdown("---")
    
    # Patient tabs
    tab1, tab2, tab3 = st.tabs(["Profile", "Prediction History", "Appointments"])
    
    with tab1:
        profile_management()
    
    with tab2:
        show_prediction_history(user['id'])
    
    with tab3:
        show_appointments(user['id'])

def show_prediction_history(user_id):
    """Show prediction history for a specific user"""
    st.header("Your Prediction History")
    
    # Get prediction history
    predictions = get_user_predictions(user_id)
    
    if not predictions:
        st.info("You haven't made any predictions yet.")
        return
    
    # Format the data for display
    formatted_data = []
    
    for pred in predictions:
        # Format confidence safely
        confidence_str = "N/A"
        if 'confidence' in pred and pred['confidence'] is not None:
            try:
                confidence_str = f"{float(pred['confidence']):.1f}%"
            except (ValueError, TypeError):
                confidence_str = "N/A"
        
        # Handle Firestore Timestamp for created_at
        created_at_str = "N/A"
        if 'created_at' in pred and isinstance(pred['created_at'], datetime):
            try:
                # Format timestamp from Firebase
                created_at_str = pred['created_at'].strftime("%Y-%m-%d %H:%M")
            except Exception:
                 created_at_str = "Invalid Date"

        formatted_data.append({
            "Date": created_at_str,
            "Type": pred.get('prediction_type', 'N/A'),
            "Result": pred.get('prediction_result', 'N/A'),
            "Confidence": confidence_str,
            "Recommendation": pred.get('recommendation', 'N/A')
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(formatted_data)
    
    # Display as table
    st.dataframe(df, use_container_width=True)
    
    # Detailed view for the most recent prediction
    if predictions: # Check if predictions list is not empty
        latest_prediction = predictions[0]
        
        # Format latest prediction confidence safely
        latest_confidence_str = "N/A"
        if 'confidence' in latest_prediction and latest_prediction['confidence'] is not None:
            try:
                latest_confidence_str = f"{float(latest_prediction['confidence']):.1f}%"
            except (ValueError, TypeError):
                latest_confidence_str = "N/A"
        
        # Handle Firestore Timestamp for created_at
        latest_created_at_str = "N/A"
        if 'created_at' in latest_prediction and isinstance(latest_prediction['created_at'], datetime):
            try:
                latest_created_at_str = latest_prediction['created_at'].strftime("%Y-%m-%d %H:%M")
            except Exception:
                 latest_created_at_str = "Invalid Date"

        st.subheader("Latest Prediction Details")
        st.markdown(f"**Date**: {latest_created_at_str}")
        st.markdown(f"**Type**: {latest_prediction.get('prediction_type', 'N/A')}")
        st.markdown(f"**Result**: {latest_prediction.get('prediction_result', 'N/A')}")
        st.markdown(f"**Confidence**: {latest_confidence_str}")
        st.markdown(f"**Recommendation**: {latest_prediction.get('recommendation', 'N/A')}")
        
        # Clinical features if available
        if 'clinical_features' in latest_prediction and latest_prediction['clinical_features']:
            st.subheader("Clinical Features Used")
            try:
                # Firestore stores dict directly, no need to json.loads
                features = latest_prediction['clinical_features']
                if isinstance(features, dict):
                    for key, value in features.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")
                else:
                     st.text(str(features)) # Display as string if not a dict
            except Exception as e:
                st.error(f"Error displaying clinical features: {e}")
        
        # Image display section removed as requested

def show_appointments(user_id):
    """Show appointments for a specific user"""
    st.header("Your Appointments")
    
    # Get appointments
    appointments = get_user_appointments(user_id)
    
    if not appointments:
        st.info("You don't have any appointments scheduled.")
        return
    
    # Format the data for display
    formatted_data = []
    
    for appt in appointments:
        # Format confidence safely
        confidence_str = "N/A"
        if 'confidence' in appt and appt['confidence'] is not None:
            try:
                confidence_str = f"{float(appt['confidence']):.1f}%"
            except (ValueError, TypeError):
                confidence_str = "N/A"
        
        # Handle appointment date (assuming it's stored as string YYYY-MM-DD)
        appt_date_str = appt.get('appointment_date', 'N/A')

        formatted_data.append({
            "Date": appt_date_str,
            "Time": appt.get('appointment_time', 'N/A'),
            "Doctor": appt.get('doctor', 'N/A'),
            "Status": appt.get('status', 'N/A').capitalize(),
            "Diagnosis": appt.get('prediction_result', 'N/A'), # Added from prediction linked
            "Confidence": confidence_str, # Added from prediction linked
            "Notes": appt.get('notes', '') # Default to empty string
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(formatted_data)
    
    # Create date filter
    today = datetime.now().strftime("%Y-%m-%d")
    date_filter = st.selectbox(
        "Filter appointments", # Renamed for clarity
        ["All", "Upcoming", "Today", "Past"]
    )
    
    filtered_df = df.copy()
    # Filter based on the string date 'YYYY-MM-DD'
    if date_filter == "Upcoming":
        filtered_df = df[df["Date"] >= today]
    elif date_filter == "Today":
        filtered_df = df[df["Date"] == today]
    elif date_filter == "Past":
        filtered_df = df[df["Date"] < today]
    
    # Display as table
    st.dataframe(filtered_df, use_container_width=True)
    
    # Upcoming appointment reminder
    # Make sure to filter based on the DataFrame's actual 'Date' column
    upcoming = df[df["Date"] >= today].sort_values(by="Date")
    if not upcoming.empty:
        next_appt = upcoming.iloc[0]
        st.success(f"""
        **Your Next Appointment**:
        - Date: {next_appt['Date']}
        - Time: {next_appt['Time']}
        - Doctor: {next_appt['Doctor']}
        
        Please arrive 15 minutes early with your ID.
        """)

# Ensure this file is not run directly if it's meant to be a module
# if __name__ == "__main__":
#     # This part might be removed if patient.py is only used as a module
#     pass 