import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the CNN model
def create_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'), 
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'), 
        layers.Flatten(),
        layers.Dense(128, activation='tanh'),
        layers.Dense(400, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Replace this with the actual shape of your input data
input_shape = (20, 20, 1)

# Load the dataset from the .npz file
dataset_dir_path = os.path.join(os.path.dirname(__file__), "dataset", "processed")
dataset_path = os.path.join(dataset_dir_path, "dataset.npz")

loaded_data = np.load(dataset_path, allow_pickle=True)
dataset = loaded_data['dataset']

# Extract input and output from the dataset
inputs = np.array([data['input'] for data in dataset])
outputs = np.array([data['output'].flatten() for data in dataset])

# Create and compile the CNN model
model = create_cnn_model(input_shape)

# Train the model
model.fit(inputs, outputs, epochs=1, batch_size=400)

# Save the model
model.save("my_model.h5")
