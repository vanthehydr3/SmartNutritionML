import csv
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from load_dataset import GroceryDataset

# Map grocery class IDs to ingredient names
class_names = {}

with open("dataset/classes.csv", "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        class_id = int(row["Coarse Class ID (int)"])
        class_name = row["Coarse Class Name (str)"]

        class_names[class_id] = class_name

print("Loaded", len(class_names), "grocery categories")

# 1. Load validation images
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

val_dataset = GroceryDataset("val.txt", transform=transform)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# 2. Recreate our fine-tuned ResNet18
model = resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    43
)

# Load our best model
model.load_state_dict(
    torch.load(
        "grocery_resnet18_finetuned.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# 3. Collect actual and predicted labels
actual_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in val_loader:

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.tolist())

# 4. Calculate confusion matrix
class_ids = list(range(43))

cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=class_ids
)

# 5. Display the confusion matrix
fig, ax = plt.subplots(figsize=(14, 12))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        class_names.get(i, str(i)) for i in class_ids
    ]
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    values_format="d",
    include_values=False,
    colorbar=False
)

ax.set_title("Fine-Tuned ResNet18 - Validation Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

# 6. Print classification metrics
print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        labels=class_ids,
        zero_division=0
    )
)

# 7. Show most common classification mistakes
mistakes = []

for actual in class_ids:
    for predicted in class_ids:
        if actual != predicted and cm[actual, predicted] > 0:
            mistakes.append(
                (cm[actual, predicted], actual, predicted)
            )

mistakes.sort(reverse=True)

print("\nMost Common Misclassifications:")

for count, actual, predicted in mistakes[:10]:
    print(
        f"Actual: {class_names.get(actual, actual)} "
        f"-> Predicted: {class_names.get(predicted, predicted)} "
        f"({count} times)"
    )

print("\nConfusion matrix saved as confusion_matrix.png")

