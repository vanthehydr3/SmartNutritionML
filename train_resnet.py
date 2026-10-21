
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from load_dataset import GroceryDataset

print("RESNET18 TRAINING STARTED", flush=True)

# 1. Use the preprocessing expected by pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# 2. Load training and validation datasets
train_dataset = GroceryDataset("train.txt", transform=transform)
val_dataset = GroceryDataset("val.txt", transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# 3. Load pretrained ResNet18
model = resnet18(weights=weights)

# Freeze the pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final layer for our 43 grocery categories
model.fc = nn.Linear(model.fc.in_features, 43)

# 4. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 5. Train
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # Evaluate using validation images
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}, "
        f"Validation Accuracy: {accuracy:.2f}%",
        flush=True
    )

# 6. Save the trained model
torch.save(model.state_dict(), "grocery_resnet18.pth")

print("Training complete! Saved grocery_resnet18.pth")
