import os
import sys

# Add project root to path to allow importing database utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import necessary functions from firebase_utils
    from database.firebase_utils import init_firebase, create_user, get_user_by_username, hash_password
    print("Successfully imported Firebase utilities.")
except ImportError as e:
    print(f"Error importing Firebase utilities: {e}")
    print("Please ensure firebase_utils.py exists in the database directory and all dependencies are installed.")
    sys.exit(1)

def setup_users():
    """Creates default admin and test users if they don't exist."""
    try:
        print("Initializing Firebase...")
        db = init_firebase()
        users_ref = db.collection('users')
        print("Firebase initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Firebase: {e}")
        print("Make sure 'firebase-key.json' is in the root directory.")
        return

    # --- Default Admin User ---
    admin_username = "admin"
    # !!! WARNING: Change this default password before running in any shared/production environment !!!
    admin_password = "admin123"
    admin_email = "admin@example.com"
    admin_full_name = "Administrator"

    print(f"Checking for existing user: {admin_username}")
    if not get_user_by_username(admin_username):
        print(f"Creating admin user: {admin_username}")
        admin_data = {
            'username': admin_username,
            'password_hash': hash_password(admin_password),
            'email': admin_email,
            'full_name': admin_full_name,
            'phone': None,
            'address': None,
            'role': 'admin', # Assign admin role
            'created_at': firestore.SERVER_TIMESTAMP
        }
        try:
            users_ref.document().set(admin_data)
            print(f"Admin user '{admin_username}' created successfully.")
        except Exception as e:
            print(f"Error creating admin user: {e}")
    else:
        print(f"Admin user '{admin_username}' already exists.")

    # --- Default Test User ---
    test_username = "test_user"
    # !!! WARNING: Change this default password !!!
    test_password = "password123"
    test_email = "test@example.com"
    test_full_name = "Test User"

    print(f"Checking for existing user: {test_username}")
    if not get_user_by_username(test_username):
        print(f"Creating test user: {test_username}")
        test_data = {
            'username': test_username,
            'password_hash': hash_password(test_password),
            'email': test_email,
            'full_name': test_full_name,
            'phone': "+1 555 123 4567",
            'address': "123 Test Street, Exampleville",
            'role': 'user', # Default role
            'created_at': firestore.SERVER_TIMESTAMP
        }
        try:
            users_ref.document().set(test_data)
            print(f"Test user '{test_username}' created successfully.")
        except Exception as e:
            print(f"Error creating test user: {e}")
    else:
        print(f"Test user '{test_username}' already exists.")

if __name__ == "__main__":
    # Need access to Firestore for the SERVER_TIMESTAMP
    try:
        from google.cloud import firestore
        print("Imported Firestore.")
        setup_users()
        print("Default user setup process completed.")
    except ImportError:
        print("Error: google-cloud-firestore is required. Please install it (`uv pip install google-cloud-firestore`)")
    except NameError as ne:
         if 'firestore' in str(ne):
              print("Error: Firestore object not available. Ensure Firebase initialization works.")
         else:
              print(f"An unexpected NameError occurred: {ne}")
    except Exception as ex:
        print(f"An unexpected error occurred during setup: {ex}") 