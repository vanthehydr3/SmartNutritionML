
from ultralytics import YOLO
from pathlib import Path

# Load our best trained YOLO checkpoint
model = YOLO(
    "runs/detect/runs/smartnutrition_yolo20/weights/best.pt"
)

# Photograph to analyze
image_path = Path("my_photos/fridge.jpg")

if not image_path.exists():
    print("Image not found:", image_path)
    exit()

# Run ingredient detection
results = model.predict(
    source=str(image_path),
    conf=0.25,
    device="mps",
    save=True
)

# Print detected ingredients
print("\nDetected Ingredients:")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        ingredient = result.names[class_id]

        print(f"{ingredient}: {confidence:.2%}")

print("\nPrediction complete!")
