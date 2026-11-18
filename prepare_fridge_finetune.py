
from pathlib import Path
import shutil

# Existing cleaned dataset
source = Path("yolo_dataset_12_clean")

# New experimental dataset
destination = Path("yolo_dataset_12_fridge")

# Refrigerator annotations
fridge = Path("fridge_training")

# Reserve fridge_001 for comparison
fridge_images = [
    f"fridge_{i:03d}.jpg"
    for i in range(2, 7)
]

# Avoid overwriting an existing experiment
if destination.exists():
    raise FileExistsError(
        f"{destination} already exists. "
        "Check the folder before proceeding."
    )

# Copy existing training and validation data
shutil.copytree(source, destination)

# Add five refrigerator examples to training
for filename in fridge_images:

    image_path = fridge / "images" / filename
    label_path = fridge / "labels" / (
        Path(filename).stem + ".txt"
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    shutil.copy2(
        image_path,
        destination / "images" / "train" / filename
    )

    shutil.copy2(
        label_path,
        destination / "labels" / "train" / label_path.name
    )

    print("Added:", filename)

# Update data.yaml to use the new folder
yaml_path = destination / "data.yaml"

yaml_text = yaml_path.read_text()
yaml_text = yaml_text.replace(
    str(source.resolve()),
    str(destination.resolve())
)

yaml_path.write_text(yaml_text)

print("\n--- FRIDGE FINE-TUNING DATASET ---")

for split in ["train", "val"]:
    image_count = len([
        p for p in (destination / "images" / split).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    label_count = len(
        list((destination / "labels" / split).glob("*.txt"))
    )

    print(
        f"{split.upper()}: "
        f"{image_count} images, "
        f"{label_count} labels"
    )

print("\nSaved to:", destination.resolve())
