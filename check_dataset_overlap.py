
from pathlib import Path
import fiftyone as fo

# Existing five-class YOLO dataset
old_dir = Path("yolo_dataset_balanced/images")

old_images = set()

for split in ["train", "val"]:
    for image_path in (old_dir / split).glob("*"):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            old_images.add(image_path.stem)

# New seven-class FiftyOne dataset
new_dataset = fo.load_dataset("smartnutrition_expanded")

new_images = {
    Path(sample.filepath).stem
    for sample in new_dataset.iter_samples()
}

overlap = old_images.intersection(new_images)

print("Original dataset images:", len(old_images))
print("New dataset images:", len(new_images))
print("Overlapping images:", len(overlap))
print("Unique combined images:", len(old_images.union(new_images)))

if overlap:
    print("\nExample overlapping image IDs:")
    for image_id in sorted(overlap)[:10]:
        print(image_id)
