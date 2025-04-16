import streamlit as st
import sys
import os

# Add the current directory to path to import database modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Switch from local database to Firebase
from database.firebase_utils import login_user, create_user, update_user_profile, create_session, get_session, delete_session

def init_auth():
    """Initialize authentication state"""
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_message' not in st.session_state:
        st.session_state.auth_message = ""
    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = ""
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
        
    # Check if we have a session cookie
    if not st.session_state.is_authenticated and 'session_id' in st.session_state and st.session_state.session_id:
        # Try to restore session
        result = get_session(st.session_state.session_id)
        if result['success']:
            st.session_state.is_authenticated = True
            st.session_state.user = result['user']

def logout():
    """Logout current user"""
    # Delete server-side session if exists
    if 'session_id' in st.session_state and st.session_state.session_id:
        delete_session(st.session_state.session_id)
        
    st.session_state.is_authenticated = False
    st.session_state.user = None
    st.session_state.session_id = None
    st.session_state.auth_message = "You have been logged out."
    st.session_state.auth_status = "info"
    st.session_state.show_signup = False

def toggle_signup():
    """Toggle between login and signup forms"""
    st.session_state.show_signup = not st.session_state.show_signup

def login_form():
    """Display login form"""
    if st.session_state.show_signup:
        return

    st.title("🔬 Skin Cancer Classification System")
    st.subheader("Login")
    
    # Show auth message if any
    if st.session_state.auth_message:
        if st.session_state.auth_status == "error":
            st.error(st.session_state.auth_message)
        elif st.session_state.auth_status == "success":
            st.success(st.session_state.auth_message)
        else:
            st.info(st.session_state.auth_message)
        
        # Clear message after displaying
        st.session_state.auth_message = ""
        st.session_state.auth_status = ""
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        remember_me = st.checkbox("Remember me")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password.")
                return
                
            result = login_user(username, password)
            if result['success']:
                st.session_state.is_authenticated = True
                st.session_state.user = result['user']
                
                # Create persistent session if remember me is checked
                if remember_me:
                    session_result = create_session(st.session_state.user['id'])
                    if session_result['success']:
                        st.session_state.session_id = session_result['session_id']
                
                st.session_state.auth_message = f"Welcome, {result['user']['full_name']}!"
                st.session_state.auth_status = "success"
                st.rerun()
            else:
                st.error(result['error'])
    
    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("Create Account"):
        st.session_state.show_signup = True
        st.rerun()

def signup_form():
    """Display signup form"""
    if not st.session_state.show_signup:
        return

    st.title("🔬 Skin Cancer Classification System")    
    st.subheader("Create Account")
    
    with st.form("signup_form"):
        username = st.text_input("Username (for login)")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        full_name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number (optional)")
        address = st.text_area("Address (optional)")
        
        submit = st.form_submit_button("Create Account")
        
        if submit:
            # Validate input
            if not username or not password or not full_name or not email:
                st.error("Please fill in all required fields.")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match.")
                return
                
            if len(password) < 8:
                st.error("Password must be at least 8 characters long.")
                return
                
            # Try to create user
            result = create_user(username, password, email, full_name, phone, address)
            if result['success']:
                st.session_state.auth_message = "Account created successfully! Please login."
                st.session_state.auth_status = "success"
                st.session_state.show_signup = False
                st.rerun()
            else:
                st.error(result['error'])
    
    st.markdown("---")
    st.markdown("Already have an account?")
    if st.button("Login"):
        st.session_state.show_signup = False
        st.rerun()

def auth_page():
    """Display the complete authentication page"""
    init_auth()
    
    if st.session_state.show_signup:
        signup_form()
    else:
        login_form()

def profile_management():
    """Display user profile management"""
    if not st.session_state.is_authenticated:
        return
        
    user = st.session_state.user
    
    st.subheader("Profile Information")
    
    with st.form("profile_form"):
        full_name = st.text_input("Full Name", value=user['full_name'])
        email = st.text_input("Email Address", value=user['email'])
        phone = st.text_input("Phone Number", value=user['phone'] if user['phone'] else "")
        address = st.text_area("Address", value=user['address'] if user['address'] else "")
        
        submit = st.form_submit_button("Update Profile")
        
        if submit:
            result = update_user_profile(user['id'], full_name, email, phone, address)
            if result['success']:
                # Update session user data
                user['full_name'] = full_name
                user['email'] = email
                user['phone'] = phone
                user['address'] = address
                st.session_state.user = user
                
                st.success("Profile updated successfully!")
            else:
                st.error(result['error'])
    
    if st.button("Logout"):
        logout()
        st.rerun()

def require_login():
    """Check if user is logged in, if not redirect to login page"""
    init_auth()
    
    if not st.session_state.is_authenticated:
        auth_page()
        st.stop()  # Stop execution here if not authenticated
        
    return st.session_state.user

def is_admin():
    """Check if current user is an admin"""
    if not st.session_state.is_authenticated:
        return False
        
    return st.session_state.user.get('is_admin', 0) == 1 