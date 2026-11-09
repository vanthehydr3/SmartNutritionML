
import fiftyone as fo
import fiftyone.zoo as foz

new_ingredients = [
    "Broccoli",
    "Bell pepper",
    "Cucumber",
    "Potato",
    "Milk",
    "Cheese",
    "Lemon"
]

images_per_class = 200

dataset_name = "smartnutrition_expanded"

# Create or load the expanded dataset
if fo.dataset_exists(dataset_name):
    dataset = fo.load_dataset(dataset_name)
    print("Loading existing expanded dataset...")
else:
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

# Download new ingredient images
for ingredient in new_ingredients:

    print(f"\nDownloading {ingredient} images...")

    temp_name = "temp_" + ingredient.lower().replace(" ", "_")

    # Reuse a temporary dataset if one already exists
    if fo.dataset_exists(temp_name):
        fo.delete_dataset(temp_name)

    subset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=[ingredient],
        max_samples=images_per_class,
        shuffle=True,
        seed=42,
        dataset_name=temp_name
    )

    # Avoid duplicate image paths
    existing_paths = set(dataset.values("filepath"))

    new_samples = [
        sample
        for sample in subset.iter_samples()
        if sample.filepath not in existing_paths
    ]

    if new_samples:
        dataset.add_samples(new_samples)

    print(f"Added {len(new_samples)} new images")

    fo.delete_dataset(temp_name)

print("\nExpanded dataset download complete!")
print(
    "Total new ingredient images:",
    sum(1 for _ in dataset.iter_samples())
)
