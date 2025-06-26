import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the updated MobileNetV2 model
model = load_model("drowsiness_mobilenetv2.keras")  

def preprocess_frame(frame):
    # Resize to 96x96 for MobileNetV2
    face = cv2.resize(frame, (96, 96))
    face = face / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)  # Add batch dimension (1, 96, 96, 3)
    return face

def predict_drowsiness(frame):
    processed = preprocess_frame(frame)
    pred = model.predict(processed, verbose=0)[0][0]
    return pred > 0.5  # Returns True if drowsy, False if alert

# Test with a dummy black frame (all zeros)
dummy_frame = np.zeros((96, 96, 3), dtype=np.uint8)  # Size updated to (96, 96, 3)

# Predict
result = predict_drowsiness(dummy_frame)

# Output result
print("Model Prediction Done.")
print("Is Drowsy:" if result else "Is Alert.")


