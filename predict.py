
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# 1. Load grocery class names
class_names = {}

with open("dataset/classes.csv", "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        class_id = int(row["Coarse Class ID (int)"])
        class_name = row["Coarse Class Name (str)"]
        class_names[class_id] = class_name

# 2. Check that an image path was provided
if len(sys.argv) < 2:
    print("Usage: python predict.py image_path")
    sys.exit()

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print("Image not found:", image_path)
    sys.exit()

# 3. Load the trained ResNet18 model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 43)

model.load_state_dict(
    torch.load(
        "grocery_resnet18_finetuned.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# 4. Prepare the image
transform = ResNet18_Weights.DEFAULT.transforms()

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# 5. Predict the ingredient
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

# 6. Get top 3 predictions
top_probabilities, top_indices = torch.topk(
    probabilities,
    k=3,
    dim=1
)

print("\nImage:", image_path)
print("\nTop 3 Predictions:")

for rank in range(3):
    class_id = top_indices[0][rank].item()
    confidence = top_probabilities[0][rank].item() * 100

    ingredient = class_names.get(class_id, f"Class {class_id}")

    print(f"{rank + 1}. {ingredient}: {confidence:.2f}%")
