import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from CNN_Model import ChestXRayCNN
from Preprocessing_data import ChestXRay14, transform

# Configuration
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Load dataset
print("\nPreparing dataset and dataloader...")
dataset = ChestXRay14(transform=transform(), limit=35000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Initialize model, loss function, optimizer
model = ChestXRayCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train():
    print("\nStarting training...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, batch in loop:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{EPOCHS}] completed. Avg Loss: {running_loss/len(dataloader):.4f}")

# Run training
if __name__ == "__main__":
    train()