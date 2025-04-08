# Sign_Speak

![Python](https://img.shields.io/badge/Python-74.4%25-blue)
![JavaScript](https://img.shields.io/badge/JavaScript-11.1%25-yellow)
![CSS](https://img.shields.io/badge/CSS-9.7%25-ff69b4)
![HTML](https://img.shields.io/badge/HTML-4.8%25-orange)

## Overview

**Sign_Speak** is an innovative project aimed at bridging the communication gap between the deaf community and those who are not familiar with sign language. Utilizing the power of machine learning and computer vision, Sign_Speak translates sign language into spoken words in real-time.

## Features

- **Real-Time Translation**: Converts sign language gestures into spoken words instantly.
- **Machine Learning**: Employs advanced machine learning models to accurately interpret various sign language gestures.
- **User-Friendly Interface**: Designed with an intuitive and accessible interface for ease of use.
- **Cross-Platform Compatibility**: Works seamlessly across different platforms and devices.

## Technologies Used

- **Python**: The backbone of the project, handling all the machine learning and backend processing.
- **JavaScript**: For interactive elements and enhancing user experience.
- **CSS**: To style the interface and ensure a visually appealing design.
- **HTML**: The foundation of the web interface, ensuring structural integrity and accessibility.

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/SignSpeak.git
cd SignSpeak
```

Set up a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, ensure to install the packages listed above.

## Configuration

The `config.py` file contains key settings for the project:

- **Dataset Paths**: Paths for custom and Kaggle datasets.
- **Model Parameters**: `NUM_CLASSES` (e.g., 36 for digits 0-9 and letters A-Z), learning rate, batch size, number of epochs.
- **Image Preprocessing**: `IMAGE_SIZE` and other parameters.
- **CLASS_MAPPING**: Maps class names (digits and letters) to indices.
- **Guide Box Coordinates**: For positioning the guide box in the webcam capture UI.

## Usage

### Running the Web Application

To start the Flask server and run the web application for live inference:

```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`. Use the on-screen guide to capture your sign, and the application will display the predicted class and associated confidence.

### Training the Model

You can train the model using either data generators or by loading all data into memory.

To train on both Custom Processed and Kaggle datasets together:

```bash
python train.py --model-type lightweight_hybrid --kaggle --custom --all-data --epochs 15 --sequence-length 1
```

**Flags**:
- `--model-type`: Choose your model architecture (e.g., optimized_cnn, pure_rnn, lightweight_hybrid, very_lightweight).
- `--kaggle` and `--custom`: Use both datasets.
- `--all-data`: Load data into memory (use for small datasets).
- `--epochs`: Set number of training epochs.
- `--sequence-length`: Set to 1 for single image classification (or higher for sequence models).

### Inference

The prediction endpoint (`/predict`) is accessible via the web interface. When an image is captured and sent to the server, the model returns:
- **Predicted Class**: The class name (letter or digit).
- **Confidence**: The probability of the predicted class.
- **Image URLs**: URLs to the captured and processed (cropped) images.

The code in `app.py` handles the prediction logic.

## Model Architectures

The project provides several model architectures implemented in `model.py`:

- **Pure RNN Model**: Uses RNN layers after flattening image data; useful for sequential data.
- **Lightweight Hybrid CNN-RNN Model**: Applies TimeDistributed CNN layers followed by RNN (GRU/LSTM) layers.
- **Very Lightweight Hybrid CNN-RNN Model**: Uses fewer filters and simpler RNN layers for resource-constrained environments.
- **Optimized CNN Model**: A pure CNN architecture with batch normalization, dropout, and global average pooling – optimized for small image datasets.

You can select the architecture via the command-line argument `--model-type` in the training script.

## Data and Dataloader

- **Custom Processed Dataset**: Expected to have a folder structure with separate subdirectories for different classes (e.g., ALPHABETS and NUMBERS).
- **Kaggle Dataset**: Expected to follow a standard directory structure where each folder represents a class.

The `dataloader.py` file includes functions like `load_data_from_directory` and `create_data_generators` which:
- Load images and labels.
- Resize and normalize images.
- One-hot encode labels.
- Optionally apply data augmentation.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the TensorFlow and Keras teams for making deep learning accessible.
- Special thanks to developers of MediaPipe for robust hand detection.
- Inspiration from various sign language recognition research projects.

---

*Empowering communication through technology.*