
from ultralytics import YOLO
from pathlib import Path

model_path = Path(
    "runs/detect/runs/detect/"
    "smartnutrition_yolo12_20/weights/best.pt"
)

model = YOLO(str(model_path))

image_path = "my_photos/fridge_bottom.jpg"

results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    device="mps",
    save=True,
    project="runs/detect",
    name="yolo12_fridge_bottom_test"
)

print("\nDetected ingredients:")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        ingredient = model.names[class_id]

        print(f"{ingredient}: {confidence:.2%}")

print("\nPrediction complete!")
