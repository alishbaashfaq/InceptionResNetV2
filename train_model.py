# train_model.py
from preprocess_resize import train_generator, test_generator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
from sklearn.metrics import classification_report

# -----------------------------
# ⚡ Paths (use proper extensions)
# -----------------------------
FINAL_MODEL_PATH = 'saved_models/hawaly_model_final.keras'       # Native Keras format
BEST_MODEL_CHECKPOINT = 'saved_models/hawaly_best_model.keras'  # Checkpoint

# -----------------------------
# ⚡ Ensure test generator does not shuffle
# -----------------------------
test_generator.shuffle = False

# -----------------------------
# ⚡ If final model exists, just load and evaluate
# -----------------------------
if os.path.exists(FINAL_MODEL_PATH):
    print("⚡ Final trained model already exists! Loading the model...")
    model = load_model(FINAL_MODEL_PATH)

    # Evaluate immediately
    loss, accuracy = model.evaluate(test_generator)
    print(f"🎉 Final Test Accuracy: {accuracy*100:.2f}%")

    # Classification report
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print("📊 Classification Report:\n", report)

    exit()  # Stop script, no training needed

# -----------------------------
# ⚡ Safety steps to limit heavy computation
# -----------------------------
SAFE_TRAIN_STEPS = 200
SAFE_VAL_STEPS = 50

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_CHECKPOINT,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

NUM_CLASSES = len(train_generator.class_indices)

# -----------------------------
# 1️⃣ Load base model (InceptionResNetV2)
# -----------------------------
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # Feature extraction

# Custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# 2️⃣ Compile (feature extraction)
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 3️⃣ Train top layers only (5 epochs)
# -----------------------------
print("🔵 Starting Stage 1 Training (5 epochs)...")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5,
    steps_per_epoch=SAFE_TRAIN_STEPS,
    validation_steps=SAFE_VAL_STEPS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# 4️⃣ Fine-tuning stage (10 epochs)
# -----------------------------
print("🟢 Starting Fine-Tuning (10 epochs)...")
base_model.trainable = True
for layer in base_model.layers[:780]:
    layer.trainable = False  # Freeze first layers

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    steps_per_epoch=SAFE_TRAIN_STEPS,
    validation_steps=SAFE_VAL_STEPS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# 5️⃣ Evaluate final model
# -----------------------------
loss, accuracy = model.evaluate(test_generator)
print(f"🎉 Final Test Accuracy: {accuracy*100:.2f}%")

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("📊 Classification Report:\n", report)

# -----------------------------
# 6️⃣ Save final trained model
# -----------------------------
model.save(FINAL_MODEL_PATH)  # Save as .keras format
print(f"✅ Final model saved at: {FINAL_MODEL_PATH}")