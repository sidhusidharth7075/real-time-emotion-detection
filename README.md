# Real-Time Emotion Detection GUI

A Python GUI application that performs real-time emotion detection using your webcam. The application uses a pre-trained Keras model to detect 7 different emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Security Note

This application uses your webcam and runs locally. It does not send or store any facial data or video externally.

## Features

- **Real-time webcam feed** with live emotion detection
- **Clean, modern GUI** built with Tkinter
- **Face detection** using OpenCV's Haar Cascade classifier
- **Emotion classification** with confidence scores
- **Start/Stop controls** for easy operation
- **Visual feedback** with color-coded confidence levels
- **Error handling** for camera and model issues

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages (see requirements.txt)

## ✅ Dataset Used

- We used the FER-2013 (Facial Expression Recognition 2013) dataset, which is widely used for emotion detection tasks.
- 📦 Source: [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

## Installation

1. **Clone or download** this project to your local machine

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify setup** (optional but recommended):
   ```bash
   python test_setup.py
   ```

## Usage

1. **Run the application**:

   ```bash
   python emotion_ui.py
   ```

2. **Using the GUI**:
   - Click "Start Detection" to begin real-time emotion detection
   - Your webcam feed will appear in the main window
   - Detected emotions will be displayed at the top with confidence scores
   - Face detection boxes will be drawn around detected faces
   - Click "Stop Detection" to stop the camera and close the application

## File Structure

```
EmotionDetectionProject/
├── emotion_ui.py          # Main GUI application
├── emotion_model.h5       # Pre-trained emotion detection model
├── requirements.txt       # Python dependencies
├── test_setup.py          # Setup verification script
├── preprocessing.ipynb    # jupyter notebook code
├── README.md              # This file
└── Datasets/              # Training and test datasets ( you can find them at Kaggle - FER2013 )
```

## How It Works

1. **Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in each frame
2. **Preprocessing**: Extracts face regions and resizes them to 48x48 pixels (grayscale)
3. **Emotion Classification**: Feeds the processed face through the pre-trained Keras model
4. **Display**: Shows the detected emotion with confidence score and visual indicators

## Troubleshooting

### Common Issues

1. **"Could not open webcam"**

   - Ensure your webcam is connected and not being used by another application
   - Try closing other applications that might be using the camera

2. **"Failed to load model"**

   - Ensure `emotion_model.h5` is in the same directory as `emotion_ui.py`
   - Check that TensorFlow is properly installed

3. **Import errors**

   - Run `pip install -r requirements.txt` to install all dependencies
   - Use `python test_setup.py` to verify your installation

4. **Poor detection accuracy**
   - Ensure good lighting conditions
   - Position your face clearly in the camera view
   - Keep your face at a reasonable distance from the camera

### Performance Tips

- Close unnecessary applications to free up system resources
- Ensure good lighting for better face detection
- Position yourself directly in front of the camera
- Keep your face clearly visible and avoid rapid movements

## Technical Details

- **Model**: Pre-trained Keras CNN model for emotion classification
- **Face Detection**: OpenCV Haar Cascade classifier
- **GUI Framework**: Tkinter with custom styling
- **Threading**: Separate thread for video processing to maintain GUI responsiveness
- **Frame Rate**: ~30 FPS for smooth real-time detection

## Dependencies

- `opencv-python`: Computer vision and face detection
- `tensorflow`: Deep learning model inference
- `numpy`: Numerical computations
- `Pillow`: Image processing for GUI display
- `tkinter`: GUI framework (usually included with Python)

## License

This project is for educational and research purposes. The emotion detection model and datasets are used for demonstration of real-time computer vision applications.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this application.

## Support

If you encounter any issues, please:

1. Check the troubleshooting section above
2. Run `python test_setup.py` to verify your setup
3. Ensure all dependencies are properly installed
4. Check that your webcam is working with other applications

## 📧 Contact

For inquiries or collaboration opportunities, reach out via:

- Email: [sidhusidharth7075@gmail.com](mailto:sidhusidharth7075@gmail.com)
- LinkedIn: [LohithSappa](https://www.linkedin.com/in/lohith-sappa-aab07629a/)

---

⭐ Don't forget to **star** this repository if you found it helpful!
