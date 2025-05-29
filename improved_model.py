import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import soundfile as sf
from scipy.io.wavfile import read
import random

def augment_audio(y, sr):
    # Add random noise
    noise = np.random.normal(0, 0.005, len(y))
    y_noisy = y + noise
    
    # Time stretching
    y_stretch = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))
    
    # Pitch shifting
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.randint(-2, 2))
    
    return [y, y_noisy, y_stretch, y_shift]

def extract_features(audio_path):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=3)
        
        # Augment audio
        augmented_versions = augment_audio(y, sr)
        
        all_features = []
        for y_aug in augmented_versions:
            # Extract features
            mfcc = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y_aug, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y_aug, sr=sr)
            zero_crossing = librosa.feature.zero_crossing_rate(y_aug)
            
            # Calculate statistics for each feature
            features = []
            for feature in [mfcc, spectral_center, chroma, zero_crossing]:
                features.extend([
                    np.mean(feature),
                    np.std(feature),
                    np.max(feature),
                    np.min(feature)
                ])
            
            all_features.append(features)
        
        return np.array(all_features)
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def create_improved_model(input_dim):
    model = models.Sequential([
        # Input layer
        layers.Dense(256, input_dim=input_dim, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def load_data():
    # Load data from both positive and negative folders
    # Use the nested directories for positive and negative folders
    positive_dir = 'positive/positive'
    negative_dir = 'negative/negative'
    
    X = []
    y = []
    
    # Process positive samples
    if os.path.exists(positive_dir):
        print(f"Processing positive samples from {positive_dir}")
        for file in os.listdir(positive_dir):
            if file.endswith('.wav'):
                audio_path = os.path.join(positive_dir, file)
                print(f"Processing {audio_path}")
                features = extract_features(audio_path)
                if features is not None:
                    X.extend(features)
                    y.extend([1] * len(features))
    else:
        print(f"Positive directory not found: {positive_dir}")
    
    # Process negative samples
    if os.path.exists(negative_dir):
        print(f"Processing negative samples from {negative_dir}")
        for file in os.listdir(negative_dir):
            if file.endswith('.wav'):
                audio_path = os.path.join(negative_dir, file)
                print(f"Processing {audio_path}")
                features = extract_features(audio_path)
                if features is not None:
                    X.extend(features)
                    y.extend([0] * len(features))
    else:
        print(f"Negative directory not found: {negative_dir}")
    
    return np.array(X), np.array(y)

def train_improved_model():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found! Please ensure positive and negative folders contain .wav files.")
        return None, None
    
    print(f"Loaded {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and compile model
    model = create_improved_model(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.2%}")
    
    return model, history, X_test, y_test

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    model, history, X_test, y_test = train_improved_model()
    if model is not None:
        # Save the model in HDF5 format
        model.save('models/improved_scream_detector.h5')
        print("Model saved successfully!")
        
        # Save the model weights
        model.save_weights('models/improved_scream_detector_weights.h5')
        print("Model weights saved successfully!")
        
        # Create a detailed evaluation report
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        classes = ['Non-Scream', 'Scream']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('models/confusion_matrix.png')
        print("Confusion matrix saved as 'models/confusion_matrix.png'")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        print("Training history saved as 'models/training_history.png'") 