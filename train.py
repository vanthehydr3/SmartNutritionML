print("TRAIN.PY HAS STARTED", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from load_dataset import GroceryDataset
from model import GroceryCNN

# 1. Prepare images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 2. Load training data
train_dataset = GroceryDataset("train.txt", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# 3. Create our CNN model
model = GroceryCNN(num_classes=43)

# 4. Define how the model learns
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train for 3 epochs
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        # Clear previous gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(images)

        # Calculate prediction error
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

# 6. Save the trained model
torch.save(model.state_dict(), "grocery_cnn.pth")

print("Training complete! Model saved as grocery_cnn.pth")
