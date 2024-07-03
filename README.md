# Fish Detection GUI

This project is a GUI application for detecting various fish species in videos using a YOLO (You Only Look Once) object detection model. The application can handle different types of inputs, including local video files, YouTube links, and live streams.

## Project Overview

The main goal of this project is to provide an easy-to-use interface for detecting fish in videos. The project pipeline includes data collection, annotation, augmentation, model training, and building a GUI for detection.

## Features

- **Fish Detection**: Detects multiple species of fish in videos.
- **Multiple Input Sources**: Supports local video files, YouTube links, and live streams.
- **User-friendly Interface**: Built with PyQt5 for ease of use.
- **Fish Class Selection**: Buttons to filter detection by specific fish classes: Bermuda, Hogfish, Sergeant, Striped, and Stingray.

## Project Pipeline

### Phase 0: Data Collection and Preparation

1. **Data Collection**: Images were sourced from Google Search, Creative Commons, and Bajan Digital Creations Incorporated.
2. **Data Annotation**: Annotated using the open-source script “labelImg” for YOLO object detection.
3. **Data Augmentation**: Augmented using the CLoDSA Library to increase diversity and reduce overfitting.

### Phase 1: Core Model (YOLO)

The annotated images were used to train a YOLOv4 model. The model was tested for accuracy using the Darknet repository based on the TensorFlow framework.

### Phase 2: Running the Model

To ensure the GUI program could run on most computers, the model was converted to YOLO-tiny. This lightweight model can be compiled within the GUI due to its small size and low computing power requirements.

### Phase 3: GUI Development

A user-friendly GUI was developed using PyQt5. The GUI allows users to select the fish species they want to detect and choose between different video input sources.

## Download `Fish_Detector.exe`

The `Fish_Detector.exe` file is too large to be hosted directly on GitHub. You can download it from the following link:

[Download Fish_Detector.exe](https://drive.google.com/file/d/1euN73YMddiG7lyQPpD7cEW0hH92Zw1fR/view?usp=drive_link)


## Acknowledgements

- **MATE organization, Microsoft Azure**
- **CLoDSA for data augmentation**
- **Darknet for YOLOv4**
- **Python, OpenCV, Flask for various utilities**
- **Google Colab & Drive as working environments**

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyQt5
- OpenCV
- Darknet

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MatthewNader2/Fish-Detection-GUI.git
    cd Fish-Detection-GUI
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the YOLOv4-tiny weights and place them in the `yolo` directory.

### Running the Application

To run the GUI, execute the following command:
```bash
python src/main.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries, please contact matthewnader2@gmail.com.
```
