
import random
import shutil
from pathlib import Path

import fiftyone as fo

# Load the downloaded dataset
dataset = fo.load_dataset("smartnutrition_openimages")

# Our five ingredient categories
classes = ["Apple", "Banana", "Orange", "Tomato", "Carrot"]
class_to_id = {name: i for i, name in enumerate(classes)}

# Output directory
output_dir = Path("yolo_dataset")

# Create YOLO folders
for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (output_dir / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# Collect all samples
samples = list(dataset.iter_samples())

# Shuffle consistently
random.seed(42)
random.shuffle(samples)

# 80% training, 20% validation
split_index = int(len(samples) * 0.8)

train_samples = samples[:split_index]
val_samples = samples[split_index:]

def export_samples(samples, split):
    image_count = 0
    box_count = 0

    for sample in samples:
        source = Path(sample.filepath)

        if not source.is_file():
            continue

        # Copy image
        destination = output_dir / "images" / split / source.name
        shutil.copy2(source, destination)

        label_path = (
            output_dir / "labels" / split / f"{source.stem}.txt"
        )

        with open(label_path, "w") as label_file:
            if sample.ground_truth is None:
                continue

            for detection in sample.ground_truth.detections:

                if detection.label not in class_to_id:
                    continue

                # Skip group boxes and depictions
                if getattr(detection, "IsGroupOf", False):
                    continue

                if getattr(detection, "IsDepiction", False):
                    continue

                # FiftyOne bounding box:
                # [x_min, y_min, width, height]
                x, y, w, h = detection.bounding_box

                # Convert to YOLO center coordinates
                center_x = x + w / 2
                center_y = y + h / 2

                class_id = class_to_id[detection.label]

                label_file.write(
                    f"{class_id} {center_x:.6f} "
                    f"{center_y:.6f} {w:.6f} {h:.6f}\n"
                )

                box_count += 1

        image_count += 1

    return image_count, box_count


train_images, train_boxes = export_samples(
    train_samples, "train"
)

val_images, val_boxes = export_samples(
    val_samples, "val"
)

# Create YOLO configuration file
yaml_content = """path: ./yolo_dataset
train: images/train
val: images/val

names:
  0: Apple
  1: Banana
  2: Orange
  3: Tomato
  4: Carrot
"""

(output_dir / "data.yaml").write_text(yaml_content)

print("\nYOLO Export Complete!")
print(f"Training images: {train_images}")
print(f"Training boxes: {train_boxes}")
print(f"Validation images: {val_images}")
print(f"Validation boxes: {val_boxes}")
print("Dataset saved to:", output_dir.resolve())
