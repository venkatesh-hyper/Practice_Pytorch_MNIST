# MNIST Digit Classifier - PyTorch + Streamlit

A deep learning-powered digit recognition system trained on the MNIST dataset using PyTorch and deployed with a Streamlit web interface. Upload your own handwritten digit images and get instant predictions!

---

## Features

- ✅ Trainable CNN model using PyTorch
- ✅ Real-time digit prediction via a Streamlit web app
- ✅ Compatible with GPU/CPU automatically
- ✅ Accepts `.jpg`, `.jpeg`, `.png` digit images
- ✅ Visualizes training loss over epochs

---

## Project Structure

```
.
├── app.py                  # Streamlit app for digit classification
├── train.py                # PyTorch model training script
├── image_classifier.pt     # Trained model weights (after training)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Model Architecture

```text
Conv2d(1, 32, 3x3) → ReLU
→ Conv2d(32, 64, 3x3) → ReLU
→ Conv2d(64, 64, 3x3) → ReLU
→ Flatten
→ Linear → Output: 10 digits
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/venkatesh-hyper/Practice_Pytorch_MNIST.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Don't have `image_classifier.pt`? No problem 👇

### 3. Train the Model

```bash
python train.py
```

The model will be saved as `image_classifier.pt`.

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📷 How to Use

1. Launch the Streamlit app
2. Upload a 28x28 grayscale image of a handwritten digit
3. Click "Predict"
4. See the predicted digit 🎉

---

## 📦 Dependencies

```
torch
torchvision
numpy
matplotlib
streamlit
Pillow
```

Install them via:

```bash
pip install torch torchvision numpy matplotlib streamlit Pillow
```

---

## 🔥 Future Ideas

- Draw-your-digit canvas
- Display prediction confidence
- Training progress bar
- Export model in ONNX format
- Deploy to Hugging Face Spaces / Streamlit Cloud

---

## 🙌 Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## 👨‍💻 Author

**Venky**  
Machine Learning Engineer & Data Scientist  
[LinkedIn](https://www.linkedin.com/in/venkatesh-ml) | [Portfolio](https://www.datascienceportfol.io/venkateshml)

---

## 📄 License

MIT License. Feel free to use, modify, and share this project.
