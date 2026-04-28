# Virtual Camera (OpenCV)

A lightweight Python-based virtual camera application built using OpenCV. The project processes live webcam input and applies real-time visual transformations, enabling the output to be used in streaming tools such as OBS.

## 🚀 Features
- Real-time webcam video capture using OpenCV
- Basic image processing filters (e.g. grayscale, edge detection, simple effects)
- Frame-by-frame processing pipeline
- Output designed for virtual camera/streaming usage (OBS-compatible workflow)

## 🛠 Tech Stack
- Python
- OpenCV
- NumPy

## 📂 Project Structure
This is a prototype-level computer vision project
Filters and processing are basic transformations rather than advanced ML/CV models
No deployment pipeline or packaged virtual camera driver implementation
🎯 Learning Goals
Understand real-time video processing
Work with OpenCV frame pipelines
Explore practical use of computer vision in streaming applications
▶️ How to Run
pip install opencv-python numpy
python main.py
📌 Future Improvements
Add GPU acceleration for smoother performance
Implement advanced filters (segmentation, background removal)
Package as virtual webcam device (OBS plugin / DirectShow integration)
