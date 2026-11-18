
from pathlib import Path
import shutil

source_dir = Path("yolo_dataset_12")
clean_dir = Path("yolo_dataset_12_clean")
exclusion_file = Path("excluded_images.txt")

# Read the filenames we want to exclude
excluded = {
    line.strip()
    for line in exclusion_file.read_text().splitlines()
    if line.strip() and not line.startswith("#")
}

print("Images marked for exclusion:", len(excluded))

# Keep the original dataset unchanged
if clean_dir.exists():
    raise FileExistsError(
        "Clean dataset folder already exists. "
        "Check it before rerunning."
    )

# Copy the original dataset
shutil.copytree(source_dir, clean_dir)

removed = 0

for split in ["train", "val"]:
    image_dir = clean_dir / "images" / split
    label_dir = clean_dir / "labels" / split

    for image_path in image_dir.iterdir():
        if image_path.name not in excluded:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        image_path.unlink()

        if label_path.exists():
            label_path.unlink()

        removed += 1
        print("Excluded:", image_path.name)

# Fix the dataset path in data.yaml
yaml_path = clean_dir / "data.yaml"

yaml_text = yaml_path.read_text()
yaml_text = yaml_text.replace(
    str(source_dir.resolve()),
    str(clean_dir.resolve())
)
yaml_path.write_text(yaml_text)

print("\n--- CLEANED DATASET ---")
print("Images removed:", removed)

for split in ["train", "val"]:
    images = [
        path for path in (clean_dir / "images" / split).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    labels = list(
        (clean_dir / "labels" / split).glob("*.txt")
    )

    print(
        f"{split.upper()}: "
        f"{len(images)} images, "
        f"{len(labels)} labels"
    )

print("Saved to:", clean_dir.resolve())
