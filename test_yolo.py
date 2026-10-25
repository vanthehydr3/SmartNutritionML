
from ultralytics import YOLO

# Load a small pretrained YOLO object detection model
model = YOLO("yolo11n.pt")

# Analyze a photograph
results = model("my_photos/fridge.jpg")

# Display all detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        name = result.names[class_id]

        print(f"{name}: {confidence:.2%}")

    # Save image with bounding boxes
    result.save(filename="fridge_detected.jpg")

print("Detection complete! Saved fridge_detected.jpg")
