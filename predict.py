# predict.py
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from preprocess_resize import train_generator   # to load class labels
import tensorflow as tf

# -----------------------------
# Load your trained model
# -----------------------------
MODEL_PATH = "saved_models/hawaly_model_final.keras"
model = load_model(MODEL_PATH)

print("✅ Model loaded successfully!")

# -----------------------------
# Get class names (same order as training)
# -----------------------------
class_indices = train_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

# -----------------------------
# Function to predict image
# -----------------------------
def predict_image(img_path):
    print(f"\n🔍 Predicting for: {img_path}")

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(299, 299))  # InceptionResNetV2 size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_resnet_v2.preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    predicted_class = idx_to_class[predicted_idx]

    print(f"🟢 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")

    return predicted_class, confidence

# -----------------------------
# Command-line argument
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict class for an image")
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()

    predict_image(args.image)
