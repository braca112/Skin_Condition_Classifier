# train.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths
DATA_DIR = "data/HAM10000"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
METADATA_CSV = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-4
RANDOM_STATE = 42

# Read metadata
df = pd.read_csv(METADATA_CSV)

# Metadata in HAM10000 uses 'image_id' and 'dx' (diagnosis code)
# create filename column
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))

# Keep only rows where file exists
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)

# Map class codes to human labels (as in HAM10000)
label_map = {
    'nv': 'melanocytic_nevus',
    'mel': 'melanoma',
    'bkl': 'benign_keratosis',
    'bcc': 'basal_cell_carcinoma',
    'akiec': 'actinic_keratoses',
    'vasc': 'vascular_lesion',
    'df': 'dermatofibroma'
}
df['label'] = df['dx'].map(label_map)

# Drop any rows without mapping (just in case)
df = df.dropna(subset=['label']).reset_index(drop=True)

# Train/val/test split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=RANDOM_STATE)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=RANDOM_STATE)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Build model (MobileNetV2 base)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
preds = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

# Freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'skin_model.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, early]
)

# Optionally unfreeze some layers and fine-tune
for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(optimizer=Adam(LR/10), loss='categorical_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(
    train_generator,
    epochs=6,
    validation_data=val_generator,
    callbacks=[checkpoint, early]
)

print("Training finished. Best model saved to models/skin_model.h5")
