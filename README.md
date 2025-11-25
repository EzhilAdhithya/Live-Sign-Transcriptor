# Live Sign Language Transcriptor

A real-time sign language recognition system that translates sign language gestures into text using deep learning and computer vision.

## Features

- Real-time hand gesture recognition using MediaPipe
- Deep learning model for sign language classification
- Support for multiple sign language alphabets
- Simple and intuitive interface
- Easy-to-use data collection tool for expanding the dataset

## Prerequisites

- Python 3.7 or higher
- OpenCV
- MediaPipe
- TensorFlow/Keras
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Live-Sign-Language-Transcriptor.git
   cd Live-Sign-Language-Transcriptor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Collection
To collect training data for new signs:
```bash
python collectdata.py
```

### Training the Model
To train the model with the collected data:
```bash
python trainmodel.py
```

### Running the Application
To start the real-time sign language translator:
```bash
python app.py
```

## Project Structure

```
Live-Sign-Language-Transcriptor/
├── .gitignore           # Specifies intentionally untracked files to ignore
├── README.md            # This file
├── app.py               # Main application for real-time sign language translation
├── collectdata.py       # Script for collecting training data
├── data.py              # Data loading and preprocessing utilities
├── function.py          # Core functions for hand detection and keypoint extraction
├── trainmodel.py        # Script for training the LSTM model
├── model.json           # Model architecture (JSON)
└── requirements.txt     # Required Python packages
```

## Model Architecture

The system uses a combination of:
- **MediaPipe Hands** for hand landmark detection
- **LSTM (Long Short-Term Memory)** neural network for sequence classification
- **Dense layers** for final prediction

The model takes a sequence of hand keypoints as input and outputs the predicted sign language character.

## Dataset

The model is trained on a custom dataset of sign language gestures. The dataset includes:
- Multiple sequences of each sign
- Frame-by-frame hand keypoint data
- Corresponding labels for each sequence

### Supported Signs
Currently, the system supports the following signs:
- Letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- Numbers: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

## Training

The model can be trained using the `trainmodel.py` script. The training process includes:
1. Loading and preprocessing the dataset
2. Splitting the data into training and validation sets
3. Training the LSTM model
4. Saving the trained model weights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

