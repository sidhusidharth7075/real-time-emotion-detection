import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import threading
import time

class EmotionDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Emotion Detection")
        self.root.geometry("800x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.cap = None
        self.is_detecting = False
        self.current_emotion = "No emotion detected"
        self.emotion_confidence = 0.0
        
        # Load model and cascade
        try:
            self.model = load_model('emotion_model.h5')
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            root.destroy()
            return
        
        # Create GUI elements
        self.create_widgets()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Real-Time Emotion Detection", 
                              font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Emotion display frame
        emotion_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        emotion_frame.pack(fill='x', pady=(0, 20))
        
        # Current emotion label
        self.emotion_label = tk.Label(emotion_frame, text="No emotion detected", 
                                     font=('Arial', 18, 'bold'), fg='#e74c3c', bg='#34495e')
        self.emotion_label.pack(pady=10)
        
        # Confidence label
        self.confidence_label = tk.Label(emotion_frame, text="Confidence: 0%", 
                                        font=('Arial', 12), fg='#ecf0f1', bg='#34495e')
        self.confidence_label.pack(pady=(0, 10))
        
        # Video frame
        video_frame = tk.Frame(main_frame, bg='#34495e', relief='sunken', bd=3)
        video_frame.pack(expand=True, fill='both', pady=(0, 20))
        
        # Video label
        self.video_label = tk.Label(video_frame, text="Click 'Start Detection' to begin", 
                                   font=('Arial', 16), fg='#bdc3c7', bg='#34495e')
        self.video_label.pack(expand=True)
        
        # Control buttons frame
        button_frame = tk.Frame(main_frame, bg='#2c3e50')
        button_frame.pack(fill='x', pady=(0, 10))
        
        # Start button
        self.start_button = tk.Button(button_frame, text="Start Detection", 
                                     command=self.start_detection,
                                     font=('Arial', 14, 'bold'),
                                     bg='#27ae60', fg='white',
                                     relief='raised', bd=3,
                                     width=15, height=2)
        self.start_button.pack(side='left', padx=(0, 10))
        
        # Stop button
        self.stop_button = tk.Button(button_frame, text="Stop Detection", 
                                    command=self.stop_detection,
                                    font=('Arial', 14, 'bold'),
                                    bg='#e74c3c', fg='white',
                                    relief='raised', bd=3,
                                    width=15, height=2,
                                    state='disabled')
        self.stop_button.pack(side='left')
        
        # Status bar
        self.status_label = tk.Label(main_frame, text="Ready", 
                                    font=('Arial', 10), fg='#95a5a6', bg='#2c3e50')
        self.status_label.pack(side='bottom', pady=(10, 0))
        
    def start_detection(self):
        """Start the emotion detection process"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam. Please check if your camera is connected.")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_detecting = True
            
            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="Detection active - Camera running")
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
            self.stop_detection()
    
    def stop_detection(self):
        """Stop the emotion detection process"""
        self.is_detecting = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Update UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.video_label.config(text="Click 'Start Detection' to begin", image='')
        self.emotion_label.config(text="No emotion detected")
        self.confidence_label.config(text="Confidence: 0%")
        self.status_label.config(text="Ready")
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.is_detecting:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, emotion, confidence = self.process_frame(frame)
                
                # Update emotion display
                self.current_emotion = emotion
                self.emotion_confidence = confidence
                
                # Convert frame for Tkinter
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize frame to fit GUI
                display_width = 600
                display_height = 450
                frame_pil = frame_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                # Update GUI in main thread
                self.root.after(0, self.update_display, frame_tk, emotion, confidence)
                
                # Control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
                break
        
        # Clean up if loop ends unexpectedly
        if self.is_detecting:
            self.root.after(0, self.stop_detection)
    
    def process_frame(self, frame):
        """Process a single frame for emotion detection"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        emotion = "No face detected"
        confidence = 0.0
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract and preprocess face
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)
            
            # Predict emotion
            try:
                prediction = self.model.predict(face_roi, verbose=0)
                emotion_index = np.argmax(prediction)
                emotion = self.emotion_labels[emotion_index].title()
                confidence = float(prediction[0][emotion_index]) * 100
                
                # Display emotion on frame
                cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                emotion = "Prediction error"
                confidence = 0.0
        
        return frame, emotion, confidence
    
    def update_display(self, frame_tk, emotion, confidence):
        """Update the GUI display with new frame and emotion data"""
        # Update video display
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk  # Keep a reference
        
        # Update emotion display
        self.emotion_label.config(text=emotion)
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Update emotion color based on confidence
        if confidence > 70:
            self.emotion_label.config(fg='#27ae60')  # Green for high confidence
        elif confidence > 40:
            self.emotion_label.config(fg='#f39c12')  # Orange for medium confidence
        else:
            self.emotion_label.config(fg='#e74c3c')  # Red for low confidence
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_detecting:
            self.stop_detection()
        self.root.destroy()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = EmotionDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 