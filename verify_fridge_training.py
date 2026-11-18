
from pathlib import Path
from collections import Counter
import math

dataset = Path("fridge_training")

classes = [
    "Apple", "Banana", "Orange", "Tomato",
    "Carrot", "Broccoli", "Bell pepper",
    "Cucumber", "Potato", "Milk",
    "Cheese", "Lemon"
]

counts = Counter()
total_boxes = 0
errors = 0

for image_path in sorted((dataset / "images").glob("*.jpg")):

    label_path = dataset / "labels" / f"{image_path.stem}.txt"

    print(f"\n{image_path.name}")

    if not label_path.exists():
        print("ERROR: Missing label file")
        errors += 1
        continue

    lines = label_path.read_text().splitlines()

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        try:
            if len(parts) != 5:
                raise ValueError("Expected 5 values")

            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])

            if not 0 <= class_id < len(classes):
                raise ValueError("Invalid class ID")

            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                raise ValueError("Non-finite coordinate")

            if not (
                0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                raise ValueError("Coordinates outside valid range")

            counts[classes[class_id]] += 1
            total_boxes += 1

        except ValueError as error:
            print(f"ERROR on line {line_number}: {error}")
            errors += 1

    print("Boxes:", len(lines))

print("\n--- FINAL VERIFICATION ---")
print("Total valid boxes:", total_boxes)
print("Errors:", errors)

print("\nIngredient distribution:")
for name in classes:
    if counts[name]:
        print(f"{name}: {counts[name]}")
