import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision.datasets import ImageFolder
from pred_image import train_with_image
from torch.utils.data import DataLoader
import torch.optim as optim
import tempfile

# Paths to dataset
train_dir = "dataset/train"
val_dir = "dataset/val"

# Define the CNN model structure
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model_path = "cnn_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # Replace this with the number of classes in your model
model = CNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Define class labels
class_labels = ["car", "dog"]  # Replace with your actual class names

# Create the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# pred
# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load datasets
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Number of classes
num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes}")

# Define CNN model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




# ################








# Route to handle prediction and training with uploaded image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            file_location = tmp_file.name
            with open(file_location, "wb") as f:
                f.write(file.file.read())

        # Open the image
        image = Image.open(file_location).convert('RGB')

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        model.eval()  # Set model to evaluation mode
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]

        # Train the model with the same image (you can choose to do this separately)
        label = 1  # Replace with the appropriate class label for the image
        train_with_image(file_location, label)

        # Clean up the temporary image file
        os.remove(file_location)

        return JSONResponse(content={"prediction": predicted_class})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to train model with a single image
def train_with_image(image_path, label):
    # Convert label to tensor
    label_tensor = torch.tensor([label]).to(device)

    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Train the model on this single image
    model.train()  # Set model to training mode
    optimizer.zero_grad()
    
    outputs = model(image_tensor)
    loss = criterion(outputs, label_tensor)
    loss.backward()
    optimizer.step()
    
    print(f"Training with image {image_path}, Loss: {loss.item():.4f}")
    
    # Save the model after training
    model_path = "cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")