from PIL import Image

from pathlib import Path
import random
import cv2

# Our ingredient classes
class_names = [
    "Apple",
    "Banana",
    "Orange",
    "Tomato",
    "Carrot"
]

# Find training images
image_dir = Path("yolo_dataset/images/train")
label_dir = Path("yolo_dataset/labels/train")

image_files = list(image_dir.glob("*.jpg"))

print("Training images found:", len(image_files))

# Pick a random image
image_path = random.choice(image_files)
label_path = label_dir / f"{image_path.stem}.txt"

# Read the image
image = cv2.imread(str(image_path))

height, width = image.shape[:2]

# Read the bounding boxes
if label_path.exists():
    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            # YOLO normalized coordinates
            center_x, center_y, box_w, box_h = map(
                float, values[1:]
            )

            # Convert normalized coordinates to pixels
            x1 = int((center_x - box_w / 2) * width)
            y1 = int((center_y - box_h / 2) * height)
            x2 = int((center_x + box_w / 2) * width)
            y2 = int((center_y + box_h / 2) * height)

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw ingredient name
            cv2.putText(
                image,
                class_names[class_id],
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# Display image
print("Showing:", image_path.name)

# Save the image with bounding boxes
output_path = "yolo_visualization.jpg"
cv2.imwrite(output_path, image)

# Open it using Windows' default image viewer
Image.open(output_path).show()

print("Visualization saved as:", output_path)