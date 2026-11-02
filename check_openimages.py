
from fiftyone.utils.openimages import get_classes

# Get the available Open Images V7 classes
classes = get_classes(version="v7")

# Ingredients we want to recognize
ingredients = [
    "Apple",
    "Banana",
    "Orange",
    "Tomato",
    "Carrot",
    "Broccoli",
    "Bell pepper",
    "Cucumber",
    "Egg",
    "Milk",
    "Cheese",
    "Potato"
]

print("Checking available grocery categories:\n")

for ingredient in ingredients:
    if ingredient in classes:
        print(f"[FOUND] {ingredient}")
    else:
        print(f"[NOT FOUND] {ingredient}")

print(f"\nTotal available classes: {len(classes)}")
