
import fiftyone as fo
from collections import Counter
from pathlib import Path

dataset = fo.load_dataset("smartnutrition_expanded")

ingredients = [
    "Broccoli",
    "Bell pepper",
    "Cucumber",
    "Potato",
    "Milk",
    "Cheese",
    "Lemon"
]

box_counts = Counter()
image_counts = Counter()

total_images = 0
usable_images = 0
empty_images = 0
missing_files = 0

for sample in dataset.iter_samples():
    total_images += 1

    if not Path(sample.filepath).is_file():
        missing_files += 1
        continue

    found_classes = set()

    if sample.ground_truth is not None:
        for detection in sample.ground_truth.detections:

            if detection.label not in ingredients:
                continue

            # Exclude group annotations and depictions
            if getattr(detection, "IsGroupOf", False):
                continue

            if getattr(detection, "IsDepiction", False):
                continue

            box_counts[detection.label] += 1
            found_classes.add(detection.label)

    if found_classes:
        usable_images += 1

        for ingredient in found_classes:
            image_counts[ingredient] += 1
    else:
        empty_images += 1

print("\n--- EXPANDED DATASET SUMMARY ---")
print("Total images:", total_images)
print("Usable images:", usable_images)
print("Images without usable labels:", empty_images)
print("Missing image files:", missing_files)

print("\n--- INGREDIENT DISTRIBUTION ---")

for ingredient in ingredients:
    print(
        f"{ingredient}: "
        f"{image_counts[ingredient]} images, "
        f"{box_counts[ingredient]} boxes"
    )

print("\nTotal usable bounding boxes:", sum(box_counts.values()))
