# Tool for Driver Alertness detection during driving

- Utilised mediapipe for face landmark detection for eye detection and ear estimation as well for yawn detection
- Uses mediapipe for nose coordinate estimate to perform PnP calculation for head pose estimation
- Uses tensorflow object detection api for mobilenet ssd v2 implementation for cell phone detection and drawing bounding boxes.


## Other technologies
- Numpy, OpenCV, Playsound (for alarm buzzing)

## Installation
- Install tensorflow-gpu
- Install tensorflow object detection api with gpu support
- Install mediapipe by python pip
    ```
    pip install mediaipipe
    ```
- Download the models from [here](https://www.google.com/)
- Run the code by 
    ```
    python run.py
    ```

# COLAB LINK
- https://colab.research.google.com/drive/1bHKkNOCDTp0fa5fpPi_gBcWkEpWgzO8E?usp=sharing

# ROBOFLOW LINK 
- https://app.roboflow.com/project-1sru8/object-detection-cell-phone/1

# Reduce Delay
    ```
        python run.py --counter 25
    ```

# Change camera
    ```
        python run.py -c 1
    ```
    or
    ```
        python run.py -c 1
    ```