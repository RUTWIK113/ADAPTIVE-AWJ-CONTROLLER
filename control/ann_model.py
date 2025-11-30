# FILE: control/ann_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os

# Define where the model is saved
MODEL_FILE = os.path.join('awj_model.keras')


def build_ann_model(input_dim=5):
    """
    Builds a DEEP Neural Network capable of learning
    complex non-linear physics curves (like 1/v relationships).
    """
    model = models.Sequential([
        # Input Layer
        layers.InputLayer(input_shape=(input_dim,)),

        # Hidden Layer 1: Wide layer to capture broad features
        layers.Dense(256, activation='relu'),

        # Hidden Layer 2: Deep layer for non-linearity
        layers.Dense(256, activation='relu'),

        # Hidden Layer 3: Narrowing down
        layers.Dense(128, activation='relu'),

        # Hidden Layer 4: Fine-tuning
        layers.Dense(64, activation='relu'),

        # Hidden Layer 5: Final processing
        layers.Dense(32, activation='relu'),

        # Output Layer: 1 neuron (Depth of Cut)
        layers.Dense(1, activation='linear')
    ])

    # Slower learning rate (0.001) for stable convergence
    optimizer = optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model