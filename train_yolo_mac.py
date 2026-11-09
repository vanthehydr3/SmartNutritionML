
from ultralytics import YOLO
from pathlib import Path

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Locate our expanded grocery dataset
dataset_path = Path(
    "yolo_dataset_balanced/data.yaml"
).resolve()

print("Starting Smart Nutrition YOLO training...")

# Train using MacBook GPU
model.train(
    data=str(dataset_path),
    epochs=20,
    imgsz=640,
    batch=4,
    device="mps",
    workers=0,
    project="runs",
    name="smartnutrition_yolo20",
    patience=10
)

print("YOLO training complete!")
