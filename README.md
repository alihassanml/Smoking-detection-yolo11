# Smoking Detection with YOLO11

This project uses a YOLO11 model to detect if a person is smoking in real-time video feeds. Built with `cv2` and `ultralytics`, this setup captures frames from a webcam, runs them through a trained YOLO11 model, and displays the detected results in real-time.

## Features
- **Real-time Detection**: Detects smoking activity in real-time through a webcam.
- **YOLO11 Model**: Utilizes a custom-trained YOLO11 model for smoking detection.
- **Annotation**: Frames are annotated with bounding boxes around detected smoking activity.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/alihassanml/Smoking-detection-yolo11.git
   cd Smoking-detection-yolo11
   ```

2. **Install Dependencies**:
   Make sure Python is installed and then install the required packages.
   ```bash
   pip install ultralytics opencv-python
   ```

3. **Download the Model**:
   Ensure you have the `best.onnx` model file in the project directory. If not, download or place your trained model in this folder.

## Usage

Run the following script to start real-time smoking detection:

```python
from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('best.onnx')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break  

    # Perform detection
    results = model(frame)
    result = results[0]

    # Annotate and display the frame
    annotated_frame = result.plot() 
    cv2.imshow('YOLO Inference', annotated_frame)
    
    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
```

## Model Training (Optional)
This project assumes you have already trained a YOLO11 model for smoking detection. For training instructions, refer to the [Ultralytics YOLO Documentation](https://github.com/ultralytics/ultralytics).

## License
This project is licensed under the MIT License.
