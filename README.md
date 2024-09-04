# Stress Detection using Facial Expressions with Deep Learning

This project aims to detect stress levels based on facial expressions using a deep learning model. The application leverages real-time video streaming to analyze a user's facial expressions and classify them into emotions such as **happy**, **neutral**, or **stressed** (based on other emotions like anger, sadness, fear, etc.). The project is built using TensorFlow, OpenCV, Flask, and other Python libraries.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

In today’s fast-paced environment, stress is a common issue that can impact well-being. Detecting stress early can be crucial in preventing health issues. This project uses facial expression analysis to detect stress in real-time via webcam input. The model has been trained on a dataset containing images of different emotions and can classify emotions into 7 categories:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

If the detected emotion is anything other than happy or neutral, the system flags it as stress.

## Features

- **Real-time Stress Detection**: Detects facial expressions from a live video feed.
- **Emotion Classification**: Classifies emotions into 7 categories.
- **Color-Coded Boxes**: Displays a green box around the face for happy/neutral emotions and a red box for stress-related emotions.
- **Web Interface**: A simple Flask-based web interface for displaying the real-time video feed and stress detection results.

  ## Dataset

This project uses the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) for training and testing the emotion recognition model. The dataset contains thousands of labeled images representing different facial expressions across various emotions:

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

Make sure to download the dataset and place it in the `data/` directory before running the training scripts or the application.

  
## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- Flask

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/stress-detection.git
    cd stress-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained model and place it in the project directory. You can download it from [link-to-model].

### Running the Application

1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Open your browser and navigate to:
    ```
    http://127.0.0.1:5000
    ```

3. The webcam feed will start, and stress detection will begin.

## Model Architecture

The model used for emotion classification is a Convolutional Neural Network (CNN) trained on a dataset of facial expressions. The architecture consists of multiple convolutional layers followed by pooling and dense layers for classification. The model was trained using the categorical crossentropy loss function and the Adam optimizer.

- **Input size**: 48x48 grayscale images
- **Output classes**: 7 emotion categories

## Usage

### Live Video Stream
Once the application is running, it will capture the video feed from your webcam. Faces in the frame are detected, and the model will predict the emotions in real-time. Depending on the emotion, the bounding box will be either green (happy/neutral) or red (stress).

### Emotion Classification
The detected emotion is displayed on the video feed above the bounding box for each face.


## Contributing

Contributions are welcome! If you have suggestions, please open an issue or a pull request. For major changes, please discuss them first by opening an issue.

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)

Special thanks to the creators of the datasets used to train the model.

