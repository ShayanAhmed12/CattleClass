"""Cattle breed classification (CS4045 Deep Learning for Perception)."""


def load_classifier(path):
    """Load a trained TensorFlow/Keras checkpoint for inference."""
    # Importing model registers the custom EfficientNet wrapper with Keras.
    from .model import keras

    return keras.models.load_model(path, compile=False)
