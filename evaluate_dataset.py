# evaluate_dataset.py

import os
import csv
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------
# 1. Load trained model
# ---------------------------------------
MODEL_PATH = r"saved_models\hawaly_model_final.keras"
model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ---------------------------------------
# 2. Dataset folder (main folder)
# ---------------------------------------
DATASET_PATH = "dataset"   # main dataset folder

# ---------------------------------------
# 3. Get all categories inside dataset/
# ---------------------------------------
class_names = sorted(os.listdir(DATASET_PATH))
print("Detected Categories:", class_names)

# ---------------------------------------
# 4. Prediction function (FIXED for 299x299)
# ---------------------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # FIXED SIZE
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array, verbose=0)
    return np.argmax(preds)  # class index


# ---------------------------------------
# CSV file setup
# ---------------------------------------
csv_file = open("evaluation_results.csv", mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

# Write CSV header
csv_writer.writerow(["Image Path", "Actual Label", "Predicted Label", "Correct/Wrong"])

# ---------------------------------------
# 5. Evaluate each category's TEST folder
# ---------------------------------------
y_true = []
y_pred = []

for class_index, class_name in enumerate(class_names):

    test_folder = os.path.join(DATASET_PATH, class_name, "test")

    if not os.path.exists(test_folder):
        print(f"No test folder found for {class_name}, skipping.")
        continue

    print(f"\nEvaluating Category: {class_name}")

    for img_file in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_file)

        # Skip non-images
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        predicted_idx = predict_image(img_path)

        y_true.append(class_index)
        y_pred.append(predicted_idx)

        status = "Correct" if predicted_idx == class_index else "Wrong"

        csv_writer.writerow([
            img_path,
            class_name,
            class_names[predicted_idx],
            status
        ])

# Close CSV file
csv_file.close()

# ---------------------------------------
# 6. Final Statistics
# ---------------------------------------
print("\n============================")
print(" FINAL EVALUATION RESULTS")
print("============================\n")

total = len(y_true)
correct = sum(np.array(y_true) == np.array(y_pred))
wrong = total - correct

print(f"Total Images: {total}")
print(f"Correct Predictions: {correct}")
print(f"Wrong Predictions: {wrong}")
print("Accuracy: {:.2f}%".format((correct / total) * 100))

# ---------------------------------------
# 7. Detailed Report
# ---------------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nCSV file saved as: evaluation_results.csv")
