# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Load the eye fixation graph images and labels
images = np.load('images.npy')
labels = np.load('labels.npy')
# Split the data into training and validation sets
train_images = images[:80]
train_labels = labels[:80]
val_images = images[80:]
val_labels = labels[80:]
base_model = keras.Sequential([
# Convolution layer with 32 filters, kernel size of 3x3, and input
shape of 224x224x3
layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224,
3)),
# Batch normalization layer
layers.BatchNormalization(),
# ReLU activation layer
layers.Activation('relu'),
# Convolution layer with 32 filters and kernel size of 3x3
layers.Conv2D(32, (3,3), activation='relu'),
# Batch normalization layer
layers.BatchNormalization(),
# ReLU activation layer
layers.Activation('relu'),
# Dropout layer with a rate of 0.25
layers.Dropout(0.25),
# Max pooling layer with pool size of 2x2
layers.MaxPooling2D((2,2)),
# Convolution layer with 64 filters and kernel size of 3x3
layers.Conv2D(64, (3,3), activation='relu'),
# Batch normalization layer
layers.BatchNormalization(),
# ReLU activation layer
layers.Activation('relu'),
# Convolution layer with 64 filters and kernel size of 3x3
layers.Conv2D(64, (3,3), activation='relu'),
# Batch normalization layer
layers.BatchNormalization(),
# ReLU activation layer
layers.Activation('relu'),
# Dropout layer with a rate of 0.25
layers.Dropout(0.25),
# Max pooling layer with pool size of 2x2
layers.MaxPooling2D((2,2)),
# Convolution layer with 128 filters and kernel size of 3x3
layers.Conv2D(128, (3,3), activation='relu'),
# Batch normalization layer
layers.BatchNormalization(),
# ReLU activation layer
layers.Activation('relu'),
# Convolution layer with 128 filters and kernel size of 3x3
layers.Conv2D(128, (3,3), activation='relu'),
# Batch normalization layer
layers.BatchNormalization(),
# ReLU activation layer
layers.Activation('relu'),
# Dropout layer with a rate of 0.25
layers.Dropout(0.25),
# Max pooling layer with pool size of 2x2
layers.MaxPooling2D((2,2)),
# Flatten layer to convert output into a 1D vector
layers.Flatten(),
# Dense layer with 64 units and ReLU activation
layers.Dense(64, activation='relu'),
# Dropout layer with a rate of 0.5
layers.Dropout(0.5),
# Output layer with a single unit and sigmoid activation
layers.Dense(1, activation='sigmoid')
])
# Freeze the layers of the pre-trained model
for layer in base_model.layers:
layer.trainable = False
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])
# Train the model
history = model.fit(train_images, train_labels, epochs=10,
validation_data=(val_images, val_labels))
