import streamlit as st
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from torch import nn

# Define the ImageClassifier class again for loading the model
class ImageClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(1,32,(3,3)),
        nn.ReLU(),
        nn.Conv2d(32,64,(3,3)),
        nn.ReLU(),
        nn.Conv2d(64,64,(3,3)),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*(28-6)*(28-6),10)
    )

  def forward(self,x):
    return self.model(x)

# Load the trained model
@st.cache_resource
def load_model():
    # Ensure the model file exists. If not, you might need to run the training code first.
    try:
        clf = ImageClassifier()
        # Load model to CPU if CUDA is not available in the Streamlit environment
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clf.load_state_dict(torch.load('image_classifier.pt', map_location=device))
        clf.to(device)
        clf.eval() # Set the model to evaluation mode
        return clf, device
    except FileNotFoundError:
        st.error("Model file 'image_classifier.pt' not found. Please run the training code to generate it.")
        return None, None

model, device = load_model()

st.title("MNIST Digit Classifier")

st.write("Upload an image of a handwritten digit (0-9) to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    # Resize to 28x28 (MNIST size) and apply transforms
    transform = ToTensor()
    image = image.resize((28, 28))
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

    if st.button("Predict"):
        with torch.no_grad():
            prediction = model(image_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()

        st.success(f"Prediction: The digit is {predicted_class}")
        probs = torch.softmax(prediction, dim=1)
    confidence = torch.max(probs).item()
    st.info(f"Confidence: {confidence*100:.2f}%")
elif model is None:
    st.warning("Model could not be loaded. Please ensure 'image_classifier.pt' exists.")

# To run this Streamlit app in Colab, you would typically:
# 1. Save this code as a Python file (e.g., `app.py`).
# 2. Run the training code first to create `image_classifier.pt`.
# 3. Install `streamlit` and `pyngrok` (`!pip install streamlit pyngrok`).
# 4. Use `!streamlit run app.py & npx localtunnel --port 8501` to run the app and expose it via a public URL.
# Due to the nature of Colab's environment, direct file execution like `!streamlit run app.py` is common.
# You'll need to make sure `image_classifier.pt` is available where you run the app.