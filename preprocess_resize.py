import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]

# -------- SETTINGS --------
IMG_SIZE = (299, 299)   # target size
BATCH_SIZE = 16
DATASET_DIR = "dataset"  # main dataset folder
# ------------------------

# Check dataset folder
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset folder '{DATASET_DIR}' not found!")

# Check that each class has 'train' and 'test' subfolders
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
for cls in classes:
    train_path = os.path.join(DATASET_DIR, cls, "train")
    test_path = os.path.join(DATASET_DIR, cls, "test")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Train or Test subfolder missing in '{cls}'")

print(f"Found classes: {classes}")

# ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Train generator: points to all 'train' subfolders inside each class
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset=None,
    shuffle=True
)

# Test generator: points to all 'test' subfolders inside each class
test_generator = test_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset=None,
    shuffle=False
)

# Print summary
print("\n--- Summary ---")
print("Classes:", train_generator.class_indices)
print("Number of training images:", train_generator.samples)
print("Number of test/validation images:", test_generator.samples)
print("Batch size:", BATCH_SIZE)
print("Image target size:", IMG_SIZE)
