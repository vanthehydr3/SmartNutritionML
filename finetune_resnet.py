
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from load_dataset import GroceryDataset

print("RESNET18 FINE-TUNING STARTED", flush=True)

# 1. Prepare data using ResNet18 preprocessing
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

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

# 2. Recreate ResNet18 architecture
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 43)

# Load our previously trained 73.99% model
model.load_state_dict(
    torch.load(
        "grocery_resnet18_baseline.pth",
        map_location="cpu",
        weights_only=True
    )
)

# 3. Freeze all model parameters
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final feature block
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Also train the classification layer
for parameter in model.fc.parameters():
    parameter.requires_grad = True

# 4. Use smaller learning rates
optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.00001},
    {"params": model.fc.parameters(), "lr": 0.0001}
])

criterion = nn.CrossEntropyLoss()

# 5. Training settings
epochs = 5

best_accuracy = 0.0
best_weights = None

for epoch in range(epochs):

    model.train()

    # Keep frozen BatchNorm layers in evaluation mode
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    total_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # 6. Evaluate on validation images
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

    # Save the best-performing weights in memory
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_weights = copy.deepcopy(model.state_dict())

# 7. Save the best fine-tuned model
torch.save(best_weights, "grocery_resnet18_finetuned.pth")

print(
    f"Fine-tuning complete! "
    f"Best validation accuracy: {best_accuracy:.2f}%"
)
print("Saved grocery_resnet18_finetuned.pth")
