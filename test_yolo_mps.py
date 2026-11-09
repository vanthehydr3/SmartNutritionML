from ultralytics import YOLO
from pathlib import Path

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Find our dataset configuration
dataset_path = Path("yolo_dataset_balanced/data.yaml").resolve()

print("Starting YOLO training on MacBook GPU...", flush=True)

# Train for one epoch
model.train(
    data=str(dataset_path),
    epochs=1,
    imgsz=640,
    batch=4,
    device="mps",
    workers=0,
    project="runs",
    name="mac_gpu_test"
)

print("GPU training test complete!")