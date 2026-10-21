from pathlib import Path
from PIL import Image

# Find all JPG images inside the training folder
image_files = list(Path("dataset/train/Fruit").rglob("*.jpg"))

print("Number of fruit images:", len(image_files))

# Open the first image
image = Image.open(image_files[0])

print("Image size:", image.size)

# Display the image
image.show()

from torchvision import transforms

# Convert image into a PyTorch tensor
transform = transforms.ToTensor()
image_tensor = transform(image.convert("RGB"))

print("Tensor shape:", image_tensor.shape)
print("First pixel values:", image_tensor[:, 0, 0])