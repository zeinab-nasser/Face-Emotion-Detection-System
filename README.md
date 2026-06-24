# Face Emotion Detection System 🧠

An end-to-end Deep Learning application built using **DeepFace** (for facial expression analysis) and **Streamlit** (for the web GUI).

## Features
- **Image Upload Mode:** Upload any facial image to get instant emotion probabilities with interactive progress bars.
- **Live Webcam Mode (Beta):** Real-time facial expression analysis with bounding boxes over the camera stream.

## Dataset & Backend Specifications
* **Dataset:** Pre-trained weights based on the **FER2013** dataset (35,887 grayscale images, 48x48 resolution, 7 emotion categories).
* **Face Detection Backend:** Built-in **OpenCV (Haar Cascades / SSD)** for rapid real-time multi-face tracking.

## How to Run
1. Activate your virtual environment:
   ```bash
   venv\Scripts\activate
## Deployed Application
https://face-emotion-detection-system-bj3acerbwlpmh53d3tnhim.streamlit.app/
