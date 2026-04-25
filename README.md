# Brain Tumour Detection using CNN

A deep learning project that detects whether a brain MRI image contains a tumour or not using a Convolutional Neural Network (CNN).

> **Note:** This project is for learning purposes only. It is not a medical diagnosis tool.

## Project Overview

This project uses MRI image classification with TensorFlow/Keras. The model is trained on two classes:

- `yes` — MRI image with tumour
- `no` — MRI image without tumour

The workflow includes:

1. Dataset download
2. Train/validation/test split
3. Image preprocessing and augmentation
4. CNN model building
5. Model training with callbacks
6. Model evaluation
7. Single-image prediction

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- Streamlit optional demo app

## Repository Structure

```text
brain-tumour-detection/
│
├── notebooks/
│   └── Brain_tumour_detection.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── model/
│   └── .gitkeep
│
├── images/
│   └── .gitkeep
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The original notebook downloads the dataset from Dropbox:

```bash
wget "https://www.dropbox.com/s/6v6p7fbs8re3max/archive.zip?dl=0"
```

Expected dataset structure after extraction:

```text
brain_tumor_dataset/
├── yes/
└── no/
```

Do not upload the full dataset to GitHub if it is large. Add a dataset link in this README instead.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumour-detection.git
cd brain-tumour-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Place your dataset in this format:

```text
brain_tumor_dataset/
├── yes/
└── no/
```

Then run:

```bash
python src/train.py --data_dir brain_tumor_dataset --epochs 30
```

The best model will be saved as:

```text
model/bestmodel.h5
```

## Predict One Image

```bash
python src/predict.py --model model/bestmodel.h5 --image path/to/image.jpg
```

Example output:

```text
Prediction score: 0.8732
Result: Tumour detected
```

## Run Streamlit App

```bash
streamlit run app.py
```

Upload an MRI image and get a prediction.

## Model Architecture

The CNN model contains:

- Conv2D layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Dropout layers to reduce overfitting
- Flatten layer to convert feature maps into vector form
- Dense layers for final classification
- Sigmoid output layer for binary classification

## Results

Add your final accuracy here after training:

```text
Test Accuracy: XX%
```

You can also add screenshots of:

- Training accuracy graph
- Validation accuracy graph
- Prediction output

## Future Improvements

- Use transfer learning with VGG16, ResNet50, or EfficientNet
- Add confusion matrix and classification report
- Deploy with Streamlit Cloud
- Improve dataset size and class balance
- Add Grad-CAM visualization to show important tumour regions

## Disclaimer

This project is for educational purposes only and should not be used as a substitute for professional medical diagnosis.
