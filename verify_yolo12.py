
from pathlib import Path
from collections import Counter

dataset_dir = Path("yolo_dataset_12")

classes = [
    "Apple", "Banana", "Orange", "Tomato",
    "Carrot", "Broccoli", "Bell pepper",
    "Cucumber", "Potato", "Milk",
    "Cheese", "Lemon"
]

class_counts = Counter()
total_boxes = 0
empty_labels = 0
invalid_boxes = 0

for split in ["train", "val"]:

    images = list(
        (dataset_dir / "images" / split).glob("*")
    )

    labels = list(
        (dataset_dir / "labels" / split).glob("*.txt")
    )

    split_boxes = 0

    for label_path in labels:
        lines = label_path.read_text().splitlines()

        if not lines:
            empty_labels += 1

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_boxes += 1
                continue

            try:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                invalid_boxes += 1
                continue

            if not (
                0 <= class_id < len(classes)
                and 0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_boxes += 1
                continue

            class_counts[classes[class_id]] += 1
            split_boxes += 1

    total_boxes += split_boxes

    print(f"\n{split.upper()}:")
    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Bounding boxes:", split_boxes)

print("\n--- FINAL DATASET VERIFICATION ---")
print("Total boxes:", total_boxes)
print("Empty label files:", empty_labels)
print("Invalid boxes:", invalid_boxes)

print("\nBoxes per category:")
for name in classes:
    print(f"{name}: {class_counts[name]}")
