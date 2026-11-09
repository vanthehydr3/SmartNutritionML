
from fiftyone.utils.openimages import get_classes

# Retrieve Open Images V7 detection classes
available_classes = set(get_classes(version="v7"))

# Existing + proposed new ingredients
ingredients = [
    "Apple",
    "Banana",
    "Orange",
    "Tomato",
    "Carrot",
    "Broccoli",
    "Bell pepper",
    "Cucumber",
    "Potato",
    "Milk",
    "Cheese",
    "Lemon"
]

print("\nSMART NUTRITION - CLASS AVAILABILITY\n")

available = []
missing = []

for ingredient in ingredients:
    if ingredient in available_classes:
        print(f"[FOUND] {ingredient}")
        available.append(ingredient)
    else:
        print(f"[NOT FOUND] {ingredient}")
        missing.append(ingredient)

print("\n--- SUMMARY ---")
print("Available categories:", len(available))
print("Unavailable categories:", len(missing))

if missing:
    print("Missing:", ", ".join(missing))
