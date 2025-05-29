import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from improved_model import extract_features, create_improved_model

def evaluate_model(model_path, test_dir):
    """
    Evaluate the model on test data and generate performance metrics
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    X_test = []
    y_test = []
    
    # Process test files
    for file in os.listdir(test_dir):
        if file.endswith('.wav'):
            # Extract label from filename (0 or 1)
            label = int(file[0])
            audio_path = os.path.join(test_dir, file)
            print(f"Processing {audio_path}")
            
            # Extract features
            features = extract_features(audio_path)
            if features is not None:
                X_test.extend(features)
                y_test.extend([label] * len(features))
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    if len(X_test) == 0:
        print("No test data found!")
        return
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred_binary)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    # Calculate accuracy
    accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Print detailed metrics
    print("\nDetailed Metrics:")
    print(f"True Negatives (Correctly identified non-screams): {cm[0,0]}")
    print(f"False Positives (Incorrectly identified as screams): {cm[0,1]}")
    print(f"False Negatives (Missed screams): {cm[1,0]}")
    print(f"True Positives (Correctly identified screams): {cm[1,1]}")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
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
    plt.savefig('models/evaluation_confusion_matrix.png')
    print("Confusion matrix saved as 'models/evaluation_confusion_matrix.png'")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('models/roc_curve.png')
    print("ROC curve saved as 'models/roc_curve.png'")
    
    return accuracy, cm, roc_auc

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Evaluate the model
    model_path = 'models/improved_scream_detector.h5'
    test_dir = 'scream_detection_main/scream_detection_main/testing'
    
    if os.path.exists(model_path):
        print(f"Evaluating model from {model_path}")
        accuracy, cm, roc_auc = evaluate_model(model_path, test_dir)
        
        # Print summary
        print("\nModel Performance Summary:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"ROC AUC: {roc_auc:.2%}")
        
        # Calculate precision, recall, and F1 score
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please run improved_model.py first to train and save the model.") 