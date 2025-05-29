import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

# Load the model
model = tf.keras.models.load_model('scream_detection_main/scream_detection_main')

# Load the test data
test_dir = 'scream_detection_main/scream_detection_main/testing'
test_files = os.listdir(test_dir)

# Prepare test data
X_test = []
y_test = []

for file in test_files:
    # Extract label from filename (0 or 1)
    label = int(file[0])
    y_test.append(label)
    
    # Load and process audio file
    audio_path = os.path.join(test_dir, file)
    # Load audio data using the same preprocessing as in modelloader.py
    from scipy.io.wavfile import read
    data, rs = read(audio_path)
    rs = rs.astype(float)
    
    # Get the required input dimension
    with open('scream_detection_main/scream_detection_main/input dimension for model.txt', 'r') as f:
        suitable_length = int(f.read())
    
    # Ensure exact length match
    if len(rs) > suitable_length:
        rs = rs[:suitable_length]
    elif len(rs) < suitable_length:
        # Pad with zeros if needed
        rs = np.pad(rs, (0, suitable_length - len(rs)))
    
    X_test.append(rs)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calculate and print metrics
print("\nModel Performance Metrics:")
print("-" * 50)
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate accuracy
accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
print(f"\nOverall Accuracy: {accuracy:.2%}")

# Print detailed metrics
print("\nDetailed Metrics:")
print(f"True Negatives (Correctly identified non-screams): {cm[0,0]}")
print(f"False Positives (Incorrectly identified as screams): {cm[0,1]}")
print(f"False Negatives (Missed screams): {cm[1,0]}")
print(f"True Positives (Correctly identified screams): {cm[1,1]}") 