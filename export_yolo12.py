
from pathlib import Path
from collections import Counter
import fiftyone as fo

# Our final 12 grocery categories
classes = [
    "Apple",
    "Banana",
    "Orange",
    "Tomato",
    "Carrot",
    "Broccoli",
    "Bell pepper",
    "Cucumber",
    "Potato",
    "Milk",
    "Cheese",
    "Lemon"
]

class_to_id = {
    name: index
    for index, name in enumerate(classes)
}

# Existing five-category YOLO dataset
old_dir = Path("yolo_dataset_balanced")

# New seven-category FiftyOne dataset
new_dataset = fo.load_dataset("smartnutrition_expanded")

# Store one record per unique image ID
images = {}

# 1. Read the original five-category dataset
for split in ["train", "val"]:

    image_dir = old_dir / "images" / split
    label_dir = old_dir / "labels" / split

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image_id = image_path.stem

        label_path = label_dir / f"{image_id}.txt"

        images[image_id] = {
            "filepath": image_path,
            "split": split,
            "boxes": []
        }

        if label_path.exists():
            with open(label_path, "r") as file:

                for line in file:
                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    class_id = int(values[0])

                    if class_id not in range(5):
                        continue

                    # Existing YOLO labels are already
                    # in normalized center coordinates
                    images[image_id]["boxes"].append(
                        (class_id, *map(float, values[1:]))
                    )

# 2. Read new seven-category annotations
new_images = 0
overlapping_images = 0

for sample in new_dataset.iter_samples():

    image_path = Path(sample.filepath)
    image_id = image_path.stem

    if image_id in images:
        overlapping_images += 1
    else:
        new_images += 1

        images[image_id] = {
            "filepath": image_path,
            "split": None,
            "boxes": []
        }

    if sample.ground_truth is None:
        continue

    for detection in sample.ground_truth.detections:

        if detection.label not in class_to_id:
            continue

        if getattr(detection, "IsGroupOf", False):
            continue

        if getattr(detection, "IsDepiction", False):
            continue

        class_id = class_to_id[detection.label]

        x, y, w, h = detection.bounding_box

        center_x = x + w / 2
        center_y = y + h / 2

        images[image_id]["boxes"].append(
            (class_id, center_x, center_y, w, h)
        )

# 3. Verify combined dataset
print("\n--- COMBINED DATASET SUMMARY ---")

print("Total unique images:", len(images))
print("New images added:", new_images)
print("Overlapping images:", overlapping_images)

print("\nClass mapping:")

for name, class_id in class_to_id.items():
    print(f"{class_id}: {name}")

# Count all currently collected annotations
counts = Counter()

for record in images.values():
    for box in record["boxes"]:
        counts[classes[box[0]]] += 1

print("\nBounding boxes by category:")

for name in classes:
    print(f"{name}: {counts[name]}")

print("\nTotal collected boxes:", sum(counts.values()))

# Count images without usable labels
empty_images = sum(
    1 for record in images.values()
    if not record["boxes"]
)

print("Images without labels:", empty_images)

print("\nNo files have been exported yet.")


import random
import shutil

# Output folder for our new 12-category dataset
output_dir = Path("yolo_dataset_12")

# Remove images without usable target annotations
usable_images = {
    image_id: record
    for image_id, record in images.items()
    if record["boxes"]
}

# Preserve original train/validation assignments
new_image_ids = [
    image_id
    for image_id, record in usable_images.items()
    if record["split"] is None
]

random.seed(42)
random.shuffle(new_image_ids)

# Assign 80% of new images to training
split_index = int(len(new_image_ids) * 0.8)

train_ids = set(new_image_ids[:split_index])
val_ids = set(new_image_ids[split_index:])

for image_id, record in usable_images.items():
    if record["split"] is None:
        record["split"] = (
            "train" if image_id in train_ids else "val"
        )

# Create output directories
for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )
    (output_dir / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# Export images and annotations

# Export images and annotations
for image_id, record in usable_images.items():

    split = record["split"]
    source = Path(record["filepath"])

    destination = (
        output_dir / "images" / split / source.name
    )

    shutil.copy2(source, destination)

    label_path = (
        output_dir / "labels" / split / f"{image_id}.txt"
    )

    # Remove duplicate boxes and validate coordinates
    unique_boxes = set()

    for box in record["boxes"]:

        class_id, cx, cy, w, h = box

        if not (0 <= class_id < len(classes)):
            continue

        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):
            continue

        if (
            cx - w / 2 < -1e-6
            or cy - h / 2 < -1e-6
            or cx + w / 2 > 1 + 1e-6
            or cy + h / 2 > 1 + 1e-6
        ):
            continue

        unique_boxes.add((
            int(class_id),
            round(cx, 6),
            round(cy, 6),
            round(w, 6),
            round(h, 6)
        ))

    # Create one label file for EACH image
    with open(label_path, "w") as file:

        for class_id, cx, cy, w, h in sorted(unique_boxes):
            file.write(
                f"{class_id} {cx} {cy} {w} {h}\n"
            )


# Create YOLO dataset configuration
yaml_content = """train: images/train
val: images/val

names:
  0: Apple
  1: Banana
  2: Orange
  3: Tomato
  4: Carrot
  5: Broccoli
  6: Bell pepper
  7: Cucumber
  8: Potato
  9: Milk
  10: Cheese
  11: Lemon
"""

# Use an absolute path for the dataset
yaml_content = (
    f"path: {output_dir.resolve()}\n" + yaml_content
)

(output_dir / "data.yaml").write_text(yaml_content)

# Verify export
print("\n--- YOLO 12-CLASS EXPORT COMPLETE ---")

for split in ["train", "val"]:

    image_count = len(
        list((output_dir / "images" / split).iterdir())
    )

    label_count = len(
        list((output_dir / "labels" / split).iterdir())
    )

    print(
        f"{split.upper()}: "
        f"{image_count} images, "
        f"{label_count} label files"
    )

print("Dataset saved to:", output_dir.resolve())
