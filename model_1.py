import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import updated_load_data
from updated_preprocess_data import ChestXRay14


# Custom collate function to handle label shape inconsistencies
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])  # Stack images
    labels = torch.stack([torch.tensor(item['label'], dtype=torch.float32) for item in batch])  # Ensure label shape
    return {'image': images, 'label': labels}


# Load dataset with transformations
transform_fn = updated_load_data.transform()
dataset = ChestXRay14(transform=transform_fn)

# Create DataLoader with custom collate function
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)

# Check a sample batch
for batch in train_loader:
    print(f"Batch image shape: {batch['image'].shape}, Batch label shape: {batch['label'].shape}")
    break


# Define the CNN Model
class ChestXRayCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(ChestXRayCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming input image is resized to 64x64
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation here since we'll use BCEWithLogitsLoss

        return x


# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 14
model = ChestXRayCNN(num_classes=num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Function
def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}/{num_epochs}...")
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}] Completed - Avg Loss: {total_loss / len(dataloader):.4f}")


# Train the model
train(model, train_loader, criterion, optimizer, device, num_epochs=10)


# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()  # Convert to numpy

            outputs = model(images)
            predictions = torch.sigmoid(outputs).cpu().numpy()

            all_labels.append(labels)
            all_preds.append(predictions)

    # Convert to numpy arrays
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Compute AUC-ROC per class
    auc_scores = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        auc_scores.append(auc)

    print(f"Mean AUC-ROC: {np.mean(auc_scores):.4f}")
    for i, auc in enumerate(auc_scores):
        print(f"Class {i + 1} AUC-ROC: {auc:.4f}")


# Evaluate the model
evaluate(model, train_loader, device)
