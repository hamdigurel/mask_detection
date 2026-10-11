# Real-Time Face Mask Detection

A webcam application that detects faces in real time and classifies each one as **with mask** or **without mask**.

## How it works

1. **Face detection**: OpenCV DNN with a pre-trained ResNet-10 SSD (Caffe) model finds faces in each frame (confidence > 0.6)
2. **Classification**: each face is resized to 224×224 and passed to a **Keras/TensorFlow** classifier (`mask_detector.model`)
3. **Visualization**: green box for "with mask", red box for "without mask"

## Installation

```bash
git clone https://github.com/hamdigurel/mask_detection.git
cd mask_detection
pip install opencv-python tensorflow numpy
```

## Usage

```bash
python main.py
```

The webcam opens automatically. Press **ESC** to exit.

## Files

```
mask_detection/
├── main.py                 # real-time detection loop
├── mask_detector.model     # trained Keras mask classifier
└── face_detector/          # OpenCV DNN face detector (prototxt + caffemodel)
```

## Tech stack

Python · OpenCV DNN · TensorFlow / Keras · NumPy
