# MusicalGestures: Gesture Recognition using Computer Vision and Artificial Intelligence applied to Musical Composition
**MusicalGestures** is an innovative application that merges hand gesture recognition with real-time audio synthesis. By using computer vision and artificial intelligence, it transforms gestures into musical notes, offering a unique and creative way to compose and perform music. This project is inspired by and builds upon [Kazuhito00's hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe).
<img src="https://github.com/user-attachments/assets/d62a79b9-6790-456e-973b-e928f084af02" width="60%">

## Features
- **Gesture-based note control**: Right-hand gestures control musical notes (C, D, E, F, G, A, B), including sharps and flats determined by thumb angles.
- **Octave selection**: Left-hand finger count adjusts the octave dynamically (ranges from 1 to 5).
- **Customizable gestures**: Train and adapt the model to recognize additional gestures.
- **Real-time performance**: High responsiveness through MediaPipe and optimized processing that benefits from having a dedicated GPU.

## How to use
- **Right Hand**: Use specific gestures to play musical notes. Sharps (`#`) and flats (`b`) are triggered based on thumb angles. Refer to the gestures below (defined in [keypoint_classifier_labels.csv](model/keypoint_classifier/keypoint_classifier_label.csv)):
  <img src="https://github.com/user-attachments/assets/82f24146-5257-4bdd-85ed-e78a43d9a16d" width="80%">
  <img src="https://github.com/user-attachments/assets/7f52005f-7254-4fb7-85e4-e717a57bde02" width="60%">
- **Left Hand**: Control the octave by extending fingers:
  - 1 finger = Octave 1
  - 2 fingers = Octave 2
  - Up to 5 fingers = Octave 5.

To run the application using your webcam do:
```bash
python main.py
```

The following options can be specified when running the application.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.7)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

# Directory
<pre>
│  main.py
│  synthesizer.py
│  hand_processing.py
│  keypoint_classification.ipynb
│
├─model
│  └─keypoint_classifier
│     │  keypoint.csv
│     │  keypoint_classifier.hdf5
│     │  keypoint_classifier.py
│     │  keypoint_classifier.tflite
│     └─ keypoint_classifier_label.csv
│
│
└─utils
    └─cvfpscalc.py
</pre>

### main.py

Program that runs the application

### synthesizer.py

Synthesizer class that is reponsible of managing the audio of the application.

### hand_processing.py
This is a sample program for inference and for adding learning data (key points) for hand sign recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

## Training
In the initial state, nine types of learning data are included: Null (class ID: 0), close hand (class ID: 1), C (class ID: 2), D (class ID:3), E (class ID:4), F (class ID:5), G (class ID:6), A (class ID:7), B (class ID:8), but It's possible to add or create your own data to retrain the model.
### Data Collection
1. Run the `hand_processing.py` script.
2. Press `k` to enable data logging mode (displayed as "MODE: Logging Key Point").
3. Use gestures and press keys `0-9` to save data into `keypoint.csv` (if you want to train new gestures delete this file first).

### Model Training
1. Open the notebook `keypoint_classification.ipynb` in Jupyter.
2. Adjust `NUM_CLASSES` to match the number of gestures.
3. Train the model and update `keypoint_classifier_label.csv` as needed.

# Reference
* [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
* [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/docs/)
* [Soundfile](https://python-soundfile.readthedocs.io/en/0.11.0/)

# License
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
