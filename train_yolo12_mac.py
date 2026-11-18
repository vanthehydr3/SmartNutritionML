
from pathlib import Path
from ultralytics import YOLO
import torch

# Confirm Apple GPU support
print("MPS available:", torch.backends.mps.is_available())

if not torch.backends.mps.is_available():
    raise RuntimeError("Apple MPS GPU is not available")

# Path to our cleaned 12-category dataset
data_path = Path(
    "yolo_dataset_12_clean/data.yaml"
).resolve()

# Start from pretrained YOLO11 nano weights
model = YOLO("yolo11n.pt")

# Train the expanded model
results = model.train(
    data=str(data_path),
    epochs=20,
    imgsz=640,
    batch=4,
    device="mps",
    workers=0,
    patience=10,
    project="runs/detect",
    name="smartnutrition_yolo12_20",
    exist_ok=False
)

print("\nTraining complete!")
print("Results saved to:", results.save_dir)
