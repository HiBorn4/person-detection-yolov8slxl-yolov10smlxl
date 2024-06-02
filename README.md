# Person Detection with YOLOv8

This repository contains a deep learning model based on YOLOv8 for efficient detection and labeling of persons in video footage. The model leverages transfer learning and GPU acceleration to achieve high accuracy and fast processing times.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Detecting and labeling persons in video footage is crucial for various applications such as surveillance, security, and analytics. Traditional methods can be slow and inaccurate. This project uses a state-of-the-art YOLOv8 model, enhanced with transfer learning, to accurately detect and label persons in video footage.

## Features
- Utilizes YOLOv8 for high-accuracy object detection.
- Transfer learning with pre-trained models for improved performance.
- Supports VGG16, ResNet50, and InceptionV3 backbones.
- GPU acceleration for faster processing.
- Generates annotated video outputs for visualization.

## Installation
1. **Clone the repository:**
    ```sh
    git clone https://github.com/HiBorn4/person-detection-yolov8.git
    cd person-detection-yolov8
    ```

2. **Create a virtual environment and activate it:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage
### Running the Model
1. **Place your video files in the `videos` folder.**

2. **Run the script to process the video:**
    ```python
    import os
    from ultralytics import YOLO
    import supervision as sv
    import numpy as np

    # Define the paths
    VIDEO_PATH = "videos/D01_20240522173959.mp4"
    MODEL_PATH = "8s/detect/train3/weights/best.pt"
    RESULT_VIDEO_PATH = "results/result.mp4"

    # Load the model
    model = YOLO(MODEL_PATH)

    # Process each frame
    def process_frame(frame: np.ndarray, _) -> np.ndarray:
        results = model(frame, imgsz=1280)[0]
        boxes = results.xyxy[:, :4].cpu().numpy()
        confidences = results.xyxy[:, 4].cpu().numpy()
        class_ids = results.xyxy[:, 5].cpu().numpy().astype(int)

        detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=2, text_scale=1.5)
        labels = [f"{model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(class_ids, confidences)]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame

    # Process the video
    sv.process_video(source_path=VIDEO_PATH, target_path=RESULT_VIDEO_PATH, callback=process_frame)
    ```

## Training the Model
To train the model from scratch or fine-tune it with your custom dataset:
```sh
yolo task=detect mode=train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=800 plots=True device=0
```

## Evaluation
To evaluate the model and check its accuracy:
```python
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Evaluate the model on the validation dataset
results = model.val(data="data.yaml")

# Print the results
print(results)
```

## Results
Here is an example of the output video with detected and labeled persons:

![Annotated Person Detection](person_detection_demo.avi)

## Contributing
We welcome contributions to improve this project. Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

Replace placeholders like `https://github.com/yourusername/person-detection-yolov8.git` and `path/to/annotated_video.gif` with actual links and paths. This README provides a comprehensive guide to understanding, installing, using, training, and evaluating the YOLOv8-based person detection model.
