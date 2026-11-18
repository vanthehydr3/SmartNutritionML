
from pathlib import Path
import random
import cv2

dataset_dir = Path("yolo_dataset_12")

classes = [
    "Apple", "Banana", "Orange", "Tomato",
    "Carrot", "Broccoli", "Bell pepper",
    "Cucumber", "Potato", "Milk",
    "Cheese", "Lemon"
]

image_dir = dataset_dir / "images" / "train"
label_dir = dataset_dir / "labels" / "train"

output_dir = Path("yolo12_visualizations")
output_dir.mkdir(exist_ok=True)

images = [
    path for path in image_dir.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

random.seed(42)
selected_images = random.sample(
    images, min(12, len(images))
)

for image_path in selected_images:

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    height, width = image.shape[:2]

    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    for line in label_path.read_text().splitlines():

        class_id, cx, cy, w, h = map(
            float, line.split()
        )

        class_id = int(class_id)

        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)

        color = (0, 255, 0)

        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            color, 2
        )

        cv2.putText(
            image,
            classes[class_id],
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    output_path = output_dir / image_path.name

    cv2.imwrite(str(output_path), image)

print("\nVisualization complete!")
print("Images saved to:", output_dir.resolve())
