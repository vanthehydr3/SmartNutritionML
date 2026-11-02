
from pathlib import Path
from ultralytics import YOLO

# Verify that our dataset configuration exists
yaml_path = Path("yolo_dataset/data.yaml").resolve()

print("Dataset configuration:", yaml_path)
print("Configuration exists:", yaml_path.is_file())

# Load our pretrained YOLO model
model = YOLO("yolo11n.pt")

# Validate that YOLO can read the dataset
try:
    model.train(
        data=str(yaml_path),
        epochs=1,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project="runs",
        name="dataset_verification"
    )

    print("\nYOLO dataset verification successful!")

except Exception as error:
    print("\nYOLO verification failed:")
    print(error)
