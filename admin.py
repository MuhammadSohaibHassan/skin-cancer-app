import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.firebase_utils import get_all_predictions, get_all_appointments
from auth import is_admin, require_login, logout

def admin_dashboard():
    """Display admin dashboard"""
    # First make sure the user is logged in
    user = require_login()
    
    # Then check if user is admin
    if not is_admin():
        st.error("You do not have permission to access this page.")
        st.stop()
    
    # Add logout button in the sidebar
    st.sidebar.title("Admin Controls")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()
    
    st.title("Admin Dashboard")
    st.markdown("---")
    
    # Admin tabs
    tab1, tab2 = st.tabs(["Prediction History", "Appointments"])
    
    with tab1:
        show_prediction_history()
    
    with tab2:
        show_appointments()

def show_prediction_history():
    """Show prediction history for all users"""
    st.header("Prediction History")
    
    # Get prediction history
    predictions = get_all_predictions()
    
    if not predictions:
        st.info("No prediction history found.")
        return
    
    # Format the data for display
    formatted_data = []
    
    for pred in predictions:
        # Parse clinical features if available
        clinical_features = None
        if 'clinical_features' in pred and pred['clinical_features']:
            # Firestore stores dict directly, no need to json.loads
            clinical_features = pred['clinical_features'] 
        
        # Format confidence safely
        confidence_str = "N/A"
        if 'confidence' in pred and pred['confidence'] is not None:
            try:
                confidence_str = f"{float(pred['confidence']):.1f}%"
            except (ValueError, TypeError):
                confidence_str = "N/A" # Keep as N/A if conversion fails
        
        # Handle Firestore Timestamp for created_at
        created_at_str = "N/A"
        if 'created_at' in pred and isinstance(pred['created_at'], datetime):
             try:
                 # Make timezone naive for consistent formatting if needed, or format directly
                 # Option 1: Format directly (assuming UTC or desired local timezone from Firestore)
                 created_at_str = pred['created_at'].strftime("%Y-%m-%d %H:%M") 
                 # Option 2: Convert to local timezone if necessary (requires pytz or similar)
                 # local_tz = pytz.timezone('Your/Local_Timezone') 
                 # created_at_local = pred['created_at'].astimezone(local_tz)
                 # created_at_str = created_at_local.strftime("%Y-%m-%d %H:%M")
             except Exception:
                 created_at_str = "Invalid Date" # Handle potential formatting errors

        formatted_data.append({
            "ID": pred.get('id', 'N/A'),
            "User": f"{pred.get('full_name', 'N/A')} ({pred.get('username', 'N/A')})",
            "Email": pred.get('email', 'N/A'),
            "Type": pred.get('prediction_type', 'N/A'),
            "Result": pred.get('prediction_result', 'N/A'),
            "Confidence": confidence_str,
            "Recommendation": pred.get('recommendation', 'N/A'),
            "Date": created_at_str, # Use the formatted string
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(formatted_data)
    
    # Display as table
    st.dataframe(df, use_container_width=True)
    
    # Export functionality
    if st.button("Export Prediction History to CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def show_appointments():
    """Show all appointments"""
    st.header("Appointments")
    
    # Get appointments
    appointments = get_all_appointments()
    
    if not appointments:
        st.info("No appointments found.")
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
            "ID": appt.get('id', 'N/A'),
            "Patient": f"{appt.get('full_name', 'N/A')} ({appt.get('username', 'N/A')})",
            "Email": appt.get('email', 'N/A'),
            "Date": appt_date_str, # Use the string date
            "Time": appt.get('appointment_time', 'N/A'),
            "Doctor": appt.get('doctor', 'N/A'),
            "Status": appt.get('status', 'N/A').capitalize(),
            "Diagnosis": appt.get('prediction_result', 'N/A'),
            "Confidence": confidence_str,
            "Notes": appt.get('notes', '') # Default to empty string
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(formatted_data)
    
    # Create date filter
    today = datetime.now().strftime("%Y-%m-%d")
    date_filter = st.selectbox(
        "Filter by date",
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
    
    # Export functionality
    if st.button("Export Appointments to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV File",
            data=csv,
            file_name=f"appointments_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    admin_dashboard() 