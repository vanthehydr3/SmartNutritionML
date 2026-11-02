
import fiftyone.zoo as foz

ingredients = [
    "Apple",
    "Banana",
    "Orange",
    "Tomato",
    "Carrot"
]

print("Starting Open Images download...")

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=ingredients,
    max_samples=100,
    shuffle=True,
    seed=42,
    dataset_name="smartnutrition_openimages"
)
# Keep the dataset available after Python exits
dataset.persistent = True

print("Dataset saved permanently!")

print("Download complete!")
print("Number of images:", len(dataset))
print("Dataset name:", dataset.name)

# Show a sample of the detection labels
sample = dataset.first()

print("\nExample image:", sample.filepath)

if sample.ground_truth is not None:
    for detection in sample.ground_truth.detections:
        print(
            "Ingredient:", detection.label,
            "Bounding box:", detection.bounding_box
        )
