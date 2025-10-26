import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Dataset path
train_dir = "dataset"

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.5)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=1,
    subset='training',
    class_mode='binary'  # Important: binary labels
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=1,
    subset='validation',
    class_mode='binary'  # Important: binary labels
)

# Simple CNN model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Single output neuron for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("mask_detector.h5")
print("✅ Model trained and saved as mask_detector.h5")
