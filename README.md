# MNIST Handwritten Digit Classification ✍️

A Streamlit web app that classifies hand-drawn digits (0–9) using a CNN
trained on the MNIST dataset.

## Live demo
[Add your Streamlit Cloud link here after deploying]

## Features
- Draw a digit directly in the browser and get an instant prediction
- Or upload an image of a digit
- Shows the model's confidence for each digit (0–9)

## Model
Trained in `Handwritten_Classification.ipynb`:
- 2 convolutional blocks (Conv2D + MaxPooling2D)
- Dense classifier head with softmax output
- Trained on the built-in `tf.keras.datasets.mnist` dataset

## Run locally
```bash
git clone https://github.com/rimiikhan2020-bot/MNIST-Handwritten-Digit-Classification.git
cd MNIST-Handwritten-Digit-Classification
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
.
├── app.py                          # Streamlit app
├── mnist_cnn.keras                 # Trained model
├── Handwritten_Classification.ipynb # Training notebook
├── requirements.txt
└── README.md
```
