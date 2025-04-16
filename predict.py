import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os

# Define the skin cancer classes
SKIN_CLASSES = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

def load_model(model_path):
    """
    Load the trained Keras model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(75, 100)):
    """
    Preprocess the image for prediction
    
    Args:
        image_path: Path to the image file
        target_size: Expected input size as (height, width)
        
    Returns:
        Preprocessed image as numpy array with shape (1, height, width, 3)
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Resize to target size - PIL uses (width, height) format
        img = img.resize((target_size[1], target_size[0]))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize image (similar to training)
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict(model, image):
    """
    Make prediction on a single image
    """
    try:
        # Reduce verbosity of TensorFlow
        tf.get_logger().setLevel('ERROR')
        
        # Make prediction
        predictions = model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        
        return {
            'class_id': int(predicted_class),
            'class_name': SKIN_CLASSES[predicted_class],
            'confidence': confidence,
            'all_probabilities': {SKIN_CLASSES[i]: float(predictions[0][i]) * 100 for i in range(len(SKIN_CLASSES))}
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    # Reduce TensorFlow verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    
    parser = argparse.ArgumentParser(description='Make skin cancer predictions using trained model')
    parser.add_argument('--model', type=str, default='skin_cancer_model.keras',
                        help='Path to the trained model file (.keras)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to predict')
    parser.add_argument('--top_n', type=int, default=3,
                        help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist.")
        return
    
    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' does not exist.")
        return
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Preprocess image
    processed_image = preprocess_image(args.image)
    if processed_image is None:
        return
    
    # Make prediction
    result = predict(model, processed_image)
    if result is None:
        return
    
    # Display results
    print("\n===== Prediction Results =====")
    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    
    print(f"\nTop {args.top_n} predictions:")
    # Sort probabilities by value (descending)
    sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, probability) in enumerate(sorted_probs[:args.top_n]):
        print(f"{i+1}. {class_name}: {probability:.2f}%")

if __name__ == "__main__":
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main() 