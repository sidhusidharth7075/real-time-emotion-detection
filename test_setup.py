#!/usr/bin/env python3
"""
Test script to verify that all dependencies and the emotion detection model are working correctly.
Run this before using the GUI to ensure everything is set up properly.
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    try:
        import tkinter as tk
        print(f"✓ Tkinter imported successfully")
    except ImportError as e:
        print(f"✗ Tkinter import failed: {e}")
        return False
    
    return True

def test_model():
    """Test if the emotion detection model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        from tensorflow.keras.models import load_model
        model = load_model('emotion_model.h5')
        print("✓ Emotion model loaded successfully")
        
        # Test model summary
        print(f"✓ Model input shape: {model.input_shape}")
        print(f"✓ Model output shape: {model.output_shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_face_cascade():
    """Test if the face cascade classifier can be loaded"""
    print("\nTesting face cascade...")
    
    try:
        import cv2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("✗ Face cascade file not found")
            return False
        else:
            print("✓ Face cascade loaded successfully")
            return True
    except Exception as e:
        print(f"✗ Face cascade loading failed: {e}")
        return False

def test_camera():
    """Test if camera can be accessed"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("✓ Camera accessible and working")
            return True
        else:
            print("✗ Camera accessible but failed to capture frame")
            return False
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Emotion Detection Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test model
    if not test_model():
        all_tests_passed = False
    
    # Test face cascade
    if not test_face_cascade():
        all_tests_passed = False
    
    # Test camera
    if not test_camera():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! You can now run the GUI with:")
        print("  python emotion_ui.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTo install dependencies, run:")
        print("  pip install -r requirements.txt")
    print("=" * 50)

if __name__ == "__main__":
    main() 