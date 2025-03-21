import torch
import torch.nn as nn
import torch.nn.functional as F

class ChestXRayCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(ChestXRayCNN, self).__init__()
        print("Initializing ChestXRayCNN model...")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Assuming input image size is 64x64 after transforms
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 64, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # -> [B, 128, 8, 8]

        x = torch.flatten(x, start_dim=1)     # -> [B, 128*8*8]
        x = F.relu(self.fc1(x))               # -> [B, 512]
        x = self.fc2(x)                       # -> [B, 14]

        return x  # No activation, use BCEWithLogitsLoss

if __name__ == "__main__":
    model = ChestXRayCNN()
    sample_input = torch.randn(4, 3, 64, 64)  # batch of 4, 3-channel 64x64 images
    output = model(sample_input)
    print("Output shape:", output.shape)  # should be [4, 14]
