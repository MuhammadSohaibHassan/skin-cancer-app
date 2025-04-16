# Skin Cancer Classification System

A web-based application using Streamlit for skin cancer classification based on clinical features analysis. Uses Firebase Firestore for data persistence.

## Features

- **Clinical Features Analysis**: Input patient clinical data for classification using a pre-trained model.
- **User Authentication**: Secure login/signup using Firebase Firestore. Supports persistent sessions ("Remember me").
- **Patient Dashboard**: View personal prediction history and manage appointments. Update profile information.
- **Admin Dashboard**: View prediction history and appointments across all users. (Accessed via `admin` user).
- **Appointment Booking**: Schedule follow-up appointments based on prediction results.
- **Report Generation**: Download a TXT report summarizing prediction results and clinical features used.

## System Requirements

- Python 3.10 specifically
- Windows, macOS, or Linux operating system
- Modern web browser (Chrome, Firefox, Edge, or Safari)
- UV package installer tool (`pip install uv`)
- Git and Git LFS (for handling the model file)

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git # Replace with your repo URL
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Set up Git LFS** (needed for the `.keras` model file):
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Create a virtual environment with Python 3.10**:
    ```bash
    # On Windows
    py -3.10 -m venv .venv
    .venv\Scripts\activate

    # On macOS/Linux
    python3.10 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install dependencies using UV**:
    ```bash
    # Ensure uv is installed: pip install uv
    uv pip install -r requirements.txt
    ```

5.  **Set up Firebase (Required)**:
    *   Follow the "Firebase Integration" steps below to create a project, enable Firestore, and get your service account key.
    *   Place the downloaded service account key file, renamed as `firebase-key.json`, in the root directory of this project. This file is ignored by Git.

6.  **Create Default Users (Optional but Recommended)**:
    *   This step creates a default `admin` and `test_user` in your Firestore database.
    *   Ensure `google-cloud-firestore` is installed: `uv pip install google-cloud-firestore` (usually included via `firebase-admin`, but good to confirm).
    *   Run the setup script from the project root directory:
        ```bash
        python setup_default_users.py
        ```
    *   *Note:* You only need to run this script once. You can also create users via the application's signup form.

## Running the Application Locally

1.  Ensure your virtual environment is active and `firebase-key.json` is present.
2.  Start the Streamlit server:
    ```bash
    streamlit run app.py
    ```
3.  The application will be available at http://localhost:8501 by default.

## Default Login Credentials (If created via setup script)

**Admin User**:
- Username: `admin`
- Password: `admin123` (Change this in `setup_default_users.py` for better security!)

**Test User**:
- Username: `test_user`
- Password: `password123` (Change this too!)

If you didn't run the setup script, use the "Create Account" button in the application.

## Firebase Integration

This application uses Firebase Firestore for database storage, which enables persistent data when deployed to Streamlit Cloud.

### Setting Up Firebase

1.  **Create a Firebase Project**:
    *   Go to the [Firebase Console](https://console.firebase.google.com/)
    *   Click "Add project" and follow the setup wizard.

2.  **Create a Firestore Database**:
    *   In the Firebase Console, go to "Firestore Database" (under Build).
    *   Click "Create database".
    *   Choose "**Start in production mode**" and select a location.

3.  **Generate Service Account Credentials**:
    *   In the Firebase Console, go to Project Settings (gear icon) > Service accounts.
    *   Select the "Firebase Admin SDK" tab.
    *   Click "**Generate new private key**" and confirm.
    *   Save the downloaded JSON file securely.

4.  **Configure Local Development**:
    *   Rename the downloaded JSON file to `firebase-key.json`.
    *   Place it in the root directory of this project.
    *   The file is already listed in `.gitignore`.

5.  **Configure Streamlit Cloud Secrets**:
    *   For deployment, copy the *entire content* of your Firebase service account JSON file.
    *   In the Streamlit Cloud dashboard for your app, go to Settings > Secrets.
    *   Paste the credentials, formatting them under the `[firebase]` key as shown in `.streamlit/secrets.toml.example`.

### Database Collections Used

-   `users`: User accounts, hashed passwords, profile data, and roles ('admin' or 'user').
-   `sessions`: Persistent authentication sessions (created when "Remember me" is checked).
-   `predictions`: Skin cancer prediction results linked to users.
-   `appointments`: Doctor appointments linked to users and predictions.

## Usage Instructions

### For Patients

1. **Login or Create Account**:
   - Use the login form or create a new account
   - Fill in your personal details during registration

2. **Navigate to Classification**:
   - Choose between image upload or clinical features input
   - For image upload, use clear, well-lit images of the lesion
   - For clinical features, provide accurate patient and lesion information

3. **View Results**:
   - See the classification results with confidence levels
   - Review the recommended actions

4. **Book Appointment**:
   - Schedule an appointment with a specialist if needed
   - Choose from available time slots

5. **Access Your Dashboard**:
   - View prediction history
   - Manage upcoming appointments
   - Update your profile information

### For Administrators

1. **Login with Admin Credentials**:
   - Use the admin username and password

2. **View Global Prediction History**:
   - See all predictions across all users
   - Export data as CSV if needed

3. **Manage Appointments**:
   - View all scheduled appointments
   - Filter by date (upcoming, past, today)
   - Export appointment data

## Project Structure

-   `app.py`: Main Streamlit application entry point.
-   `auth.py`: Handles user authentication, login/signup forms, session management.
-   `admin.py`: Defines the admin dashboard interface.
-   `patient.py`: Defines the patient dashboard interface.
-   `database/`: Contains database interaction logic.
    -   `firebase_utils.py`: Functions for interacting with Firebase Firestore (users, predictions, appointments, sessions).
-   `prediction.py`: Contains the `predict` function using the clinical features model.
-   `predict.py`: Contains image loading/prediction logic (currently image classification is separate).
-   `model/`: Likely contains model-related files (preprocessing objects).
-   `skin_cancer_model.keras`: The trained Keras model file (handled by Git LFS).
-   `metadata.csv`: Data used for feature ranges/validation.
-   `requirements.txt`: Python package dependencies.
-   `packages.txt`: System-level dependencies for Streamlit Cloud.
-   `.streamlit/`: Streamlit configuration files.
    -   `config.toml`: Theme and server settings.
    -   `secrets.toml.example`: Template for secrets (including Firebase).
-   `.gitignore`: Specifies intentionally untracked files for Git.
-   `.gitattributes`: Configures Git LFS for large files.
-   `README.md`: This file.
-   `setup_default_users.py`: Script to create default admin/test users in Firebase.

## Troubleshooting

-   **Firebase Connection Errors**:
    *   Local: Ensure `firebase-key.json` is present in the project root and correctly named.
    *   Streamlit Cloud: Double-check that the Firebase credentials in Streamlit Secrets match the format in `.streamlit/secrets.toml.example` and the content of your key file exactly. Ensure the `[firebase]` section exists.
    *   Check Firestore rules in the Firebase Console (they should allow authenticated reads/writes based on user ID for relevant collections).
-   **Model Loading Errors**:
    *   Ensure `skin_cancer_model.keras` is present (run `git lfs pull` if needed).
    *   Check that TensorFlow/Keras versions in `requirements.txt` are compatible.
-   **Dependency Issues**: Run `uv pip install -r requirements.txt` in your activated virtual environment.
-   **Authentication Issues**: If default users weren't created, run `python setup_default_users.py`. Ensure passwords match.
-   **Environment Issues**: Confirm you are using Python 3.10 specifically.

## Application Security

-   Passwords stored as SHA-256 hashes in Firestore.
-   Firebase security rules should be configured for production to restrict data access appropriately (e.g., users can only read/write their own data).
-   Session management via secure UUIDs stored in Firestore.
-   Streamlit secrets used for sensitive credentials deployment.

## Deploying to GitHub and Streamlit Cloud

### GitHub Deployment

1. **Create a GitHub Repository**:
   - Go to [GitHub](https://github.com) and sign in
   - Click the "+" icon in the top right corner and select "New repository"
   - Name your repository (e.g., "skin-cancer-classification")
   - Select "Public" or "Private" depending on your needs
   - Click "Create repository"

2. **Initialize Git and Push Your Code**:
   ```bash
   # Initialize Git in your project directory (if not already initialized)
   git init
   
   # Add all files to Git (excluding those in .gitignore)
   git add .
   
   # Commit the files
   git commit -m "Initial commit"
   
   # Add your GitHub repository as remote
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   
   # Push to GitHub
   git push -u origin main
   ```

3. **Set Up Git LFS for Large Files**:
   
   Since the model file is large (43MB), you should use Git Large File Storage (LFS):
   ```bash
   # Install Git LFS (if not already installed)
   # For Windows: Download from https://git-lfs.github.com/
   # For macOS: brew install git-lfs
   # For Linux: apt-get install git-lfs
   
   # Initialize Git LFS
   git lfs install
   
   # Track large model files
   git lfs track "*.keras"
   git lfs track "*.h5"
   
   # Add .gitattributes
   git add .gitattributes
   git commit -m "Configure Git LFS for model files"
   git push
   ```

### Streamlit Cloud Deployment

1. **Sign Up/In**: Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your GitHub account.
2. **Deploy App**: Click "New app", select repository, branch (`main`), and main file (`app.py`).
3. **Configure Secrets**: Go to app Settings > Secrets. Add your Firebase service account credentials under the `[firebase]` key, following the structure in `.streamlit/secrets.toml.example`.
4. **Advanced Settings**: Ensure Python version is set to 3.10. `requirements.txt` and `packages.txt` will be used automatically.
5. Click "Deploy".

## Notes on Streamlit Cloud Limitations

- **Resource Limits**: Streamlit Cloud has memory and computation limits
- **Public Access**: Apps on the free tier are publicly accessible
- **Inactive Apps**: Apps that don't receive traffic may be spun down
- **Storage**: No persistent file storage (use databases or cloud storage for persistent data) "# skin-cancer-app" 
