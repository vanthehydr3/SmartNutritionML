
import torch
import torch.nn as nn


class GroceryCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()

        # Learn visual features from grocery images
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Convert learned features into grocery predictions
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Test the model with a fake batch of images
if __name__ == "__main__":
    model = GroceryCNN()

    dummy_images = torch.randn(4, 3, 224, 224)
    predictions = model(dummy_images)

    print("Model output shape:", predictions.shape)
