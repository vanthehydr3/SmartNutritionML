from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GroceryDataset(Dataset):

    def __init__(self, label_file, transform=None):
        self.samples = []
        self.transform = transform

        dataset_path = Path("dataset")

        with open(dataset_path / label_file, "r") as file:
            for line in file:
                image_path, fine_label, coarse_label = line.strip().split(",")

                self.samples.append((
                    dataset_path / image_path.strip(),
                    int(coarse_label.strip())
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Only runs when load_dataset.py is executed directly
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = GroceryDataset("train.txt", transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    images, labels = next(iter(train_loader))

    print("Total training images:", len(train_dataset))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("First five labels:", labels[:5])

    unique_labels = sorted(set(label for _, label in train_dataset.samples))

    print("Number of classes:", len(unique_labels))
    print("Class IDs:", unique_labels)