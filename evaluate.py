
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from load_dataset import GroceryDataset
from model import GroceryCNN

# Prepare test images the same way as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the unseen test dataset
test_dataset = GroceryDataset("val.txt", transform=transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Recreate our CNN
model = GroceryCNN(num_classes=43)

# Load the weights learned during training
model.load_state_dict(
    torch.load("grocery_cnn.pth", map_location="cpu", weights_only=True)
)

model.eval()

correct = 0
total = 0

# Evaluate without updating model weights
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print("Total test images:", total)
print("Correct predictions:", correct)
print(f"Test accuracy: {accuracy:.2f}%")
