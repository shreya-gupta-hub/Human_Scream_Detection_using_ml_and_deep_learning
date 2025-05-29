import os
import numpy as np
import tensorflow as tf
import librosa
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window

# Set window size
Window.size = (800, 600)

class ScreamDetectorApp(App):
    def build(self):
        # Create the main layout
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Create status label
        self.status_label = Label(
            text='Scream Detector Ready',
            size_hint_y=0.2,
            font_size='20sp'
        )
        
        # Create file chooser button
        self.file_button = Button(
            text='Select Audio File',
            size_hint_y=0.2,
            background_color=(0.2, 0.6, 0.2, 1)
        )
        self.file_button.bind(on_press=self.show_file_chooser)
        
        # Create detect button
        self.detect_button = Button(
            text='Detect Scream',
            size_hint_y=0.2,
            background_color=(0.2, 0.6, 0.8, 1),
            disabled=True
        )
        self.detect_button.bind(on_press=self.detect_scream)
        
        # Add widgets to layout
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.file_button)
        self.layout.add_widget(self.detect_button)
        
        # Initialize variables
        self.selected_file = None
        
        # Load the model
        try:
            self.model = tf.keras.models.load_model('models/improved_scream_detector.h5')
            self.status_label.text = 'Model loaded successfully'
        except Exception as e:
            self.status_label.text = f'Error loading model: {str(e)}'
            self.status_label.color = (1, 0, 0, 1)  # Red color for error
        
        return self.layout
    
    def show_file_chooser(self, instance):
        # Create file chooser popup
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserListView(path=os.getcwd(), filters=['*.wav'])
        content.add_widget(file_chooser)
        
        # Create buttons
        button_layout = BoxLayout(size_hint_y=0.1)
        cancel_button = Button(text='Cancel')
        select_button = Button(text='Select')
        
        button_layout.add_widget(cancel_button)
        button_layout.add_widget(select_button)
        content.add_widget(button_layout)
        
        # Create popup
        popup = Popup(title='Select Audio File', content=content, size_hint=(0.9, 0.9))
        
        # Bind buttons
        cancel_button.bind(on_press=popup.dismiss)
        select_button.bind(on_press=lambda x: self.select_file(file_chooser.selection, popup))
        
        popup.open()
    
    def select_file(self, selection, popup):
        if selection:
            self.selected_file = selection[0]
            self.status_label.text = f'Selected: {os.path.basename(self.selected_file)}'
            self.detect_button.disabled = False
        popup.dismiss()
    
    def detect_scream(self, instance):
        if not self.selected_file:
            self.status_label.text = 'No file selected'
            return
        
        self.status_label.text = 'Processing...'
        self.detect_button.disabled = True
        
        try:
            # Extract features
            features = self.extract_features(self.selected_file)
            if features is not None:
                # Make prediction
                prediction = self.model.predict(features)
                probability = np.mean(prediction)
                
                # Update status
                if probability > 0.5:
                    # Determine risk level based on probability
                    if probability > 0.8:
                        risk_level = "HIGH RISK"
                        color = (1, 0, 0, 1)  # Red for high risk
                    elif probability > 0.65:
                        risk_level = "MEDIUM RISK"
                        color = (1, 0.5, 0, 1)  # Orange for medium risk
                    else:
                        risk_level = "LOW RISK"
                        color = (1, 0.8, 0, 1)  # Yellow for low risk
                    
                    self.status_label.text = f'Scream Detected! ({risk_level})'
                    self.status_label.color = color
                else:
                    # For non-scream, show the opposite risk level
                    if probability < 0.2:
                        risk_level = "LOW RISK"
                        color = (0, 1, 0, 1)  # Green for low risk
                    elif probability < 0.35:
                        risk_level = "MEDIUM RISK"
                        color = (0, 0.8, 0, 1)  # Light green for medium risk
                    else:
                        risk_level = "HIGH RISK"
                        color = (0, 0.5, 0, 1)  # Dark green for high risk
                    
                    self.status_label.text = f'No Scream Detected ({risk_level})'
                    self.status_label.color = color
            else:
                self.status_label.text = 'Error processing audio'
                self.status_label.color = (1, 1, 0, 1)  # Yellow color for error
        except Exception as e:
            self.status_label.text = f'Error: {str(e)}'
            self.status_label.color = (1, 0, 0, 1)  # Red color for error
        
        self.detect_button.disabled = False
    
    def extract_features(self, audio_path):
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=3)
            
            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            zero_crossing = librosa.feature.zero_crossing_rate(y)
            
            # Calculate statistics for each feature
            features = []
            for feature in [mfcc, spectral_center, chroma, zero_crossing]:
                features.extend([
                    np.mean(feature),
                    np.std(feature),
                    np.max(feature),
                    np.min(feature)
                ])
            
            # Reshape for model input
            features = np.array(features).reshape(1, -1)
            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

if __name__ == '__main__':
    ScreamDetectorApp().run() 