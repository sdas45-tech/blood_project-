import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# 🔥 YOUR DATASET PATH
DATA_DIR = Path(r"C:\Users\Sibam Das\Downloads\periodic blood image\augmented_highres")

# ==========================
# CHECK PATH
# ==========================
if not DATA_DIR.exists():
    raise FileNotFoundError(f"❌ Dataset not found: {DATA_DIR}")

print("Dataset Path:", DATA_DIR)

# ==========================
# LOAD DATA (AUTO SPLIT)
# ==========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# ==========================
# NORMALIZATION
# ==========================
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ==========================
# DATA AUGMENTATION
# ==========================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.2),
])

# ==========================
# MODEL
# ==========================
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# ==========================
# BUILD MODEL
# ==========================
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.6)(x)

x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.6)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ==========================
# COMPILE
# ==========================
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

# ==========================
# CALLBACKS
# ==========================
os.makedirs("models", exist_ok=True)

callbacks = [
    ModelCheckpoint("models/vgg16_best.keras",
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1),

    EarlyStopping(monitor="val_loss",
                  patience=5,
                  restore_best_weights=True),

    ReduceLROnPlateau(monitor="val_loss",
                      factor=0.3,
                      patience=2,
                      min_lr=1e-6,
                      verbose=1)
]

# ==========================
# TRAIN
# ==========================
print("\n🚀 Training...\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================
# EVALUATE
# ==========================
loss, acc = model.evaluate(val_ds)
print(f"\n✅ Final Accuracy: {acc*100:.2f}%")

# ==========================
# SAVE CLASS NAMES
# ==========================
class_names_path = Path("models") / "class_names.json"
with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"✅ Class names saved to {class_names_path}")
# ✅ Final Accuracy: 97.47%