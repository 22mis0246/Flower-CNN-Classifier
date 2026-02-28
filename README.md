🌸 Flower Classification using CNN

A Deep Learning project that classifies flower images into five categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

📌 Project Overview

This project implements an image classification system that identifies flowers as:

🌼 Lilly

🌸 Lotus

🌺 Orchid

🌻 Sunflower

🌷 Tulip

The model was trained using a custom dataset and achieves 81.7% validation accuracy.

🧠 Technologies Used

Python

TensorFlow

Keras

NumPy

Matplotlib

Pillow (PIL)

Git & GitHub

📂 Project Structure
flower_images/
│
├── dataset/              # Training dataset (5 flower folders)
│   ├── Lilly/
│   ├── Lotus/
│   ├── Orchid/
│   ├── Sunflower/
│   └── Tulip/
│
├── test_images/          # Images for prediction testing
│
├── train.py              # Model training script
├── predict.py            # Image prediction script
├── flower_model.h5       # Trained model file (not uploaded if large)
├── .gitignore
└── README.md
⚙️ Installation & Setup (From Scratch)
1️⃣ Clone Repository
git clone https://github.com/yourusername/Flower-CNN-Classifier.git
cd Flower-CNN-Classifier
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install tensorflow
pip install numpy
pip install matplotlib
pip install pillow
🚀 How to Train the Model
python train.py

This will:

Load dataset

Train CNN model

Save trained model as flower_model.h5

🔍 How to Predict Images

Place test images inside:

test_images/

Then run:

python predict.py

Example output:

Image: image1.jpg
Prediction: Sunflower
Confidence: 99.98%
📊 Model Performance

Training Accuracy: 98.43%

Validation Accuracy: 81.70%

📌 How It Works

Images are resized to 224x224

Pixel values normalized (0–1)

CNN extracts features using Conv2D layers

MaxPooling reduces dimensions

Dense layers classify into 5 categories

Softmax outputs probability scores

🔮 Future Improvements

Apply Data Augmentation

Use Transfer Learning (MobileNet / ResNet)

Add Dropout to reduce overfitting

Deploy as Web App

👨‍💻 Author

Naveen Raj

⭐ If you like this project

Give it a star ⭐ on GitHub
