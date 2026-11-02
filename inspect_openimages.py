
import fiftyone as fo
from collections import Counter

# Load the dataset we already downloaded
dataset = fo.load_dataset("smartnutrition_openimages")

ingredients = [
    "Apple",
    "Banana",
    "Orange",
    "Tomato",
    "Carrot"
]

# Count bounding-box annotations for each ingredient
counts = Counter()

for sample in dataset:
    if sample.ground_truth is not None:
        for detection in sample.ground_truth.detections:
            if detection.label in ingredients:
                counts[detection.label] += 1

print("Total images:", len(dataset))
print("Dataset name:", dataset.name)
print("Dataset sample count:", dataset.count())
print("First sample:", dataset.first())
print("\nIngredient annotation counts:")

for ingredient in ingredients:
    print(f"{ingredient}: {counts[ingredient]}")

print("\nTotal ingredient bounding boxes:", sum(counts.values()))

# Count samples by iterating through the dataset
actual_count = 0

for sample in dataset.iter_samples():
    actual_count += 1

print("\nVerification:")
print("Images counted manually:", actual_count)
print("Dataset length:", len(dataset))
print("Dataset count:", dataset.count())

from pathlib import Path

existing_images = 0
missing_images = 0

for sample in dataset.iter_samples():
    if Path(sample.filepath).is_file():
        existing_images += 1
    else:
        missing_images += 1

print("\nImage File Verification:")
print("Existing images:", existing_images)
print("Missing images:", missing_images)