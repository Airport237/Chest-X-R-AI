from torch.utils.data import DataLoader
import torch.nn as nn
from preprocess_data import ChestXRay14
from load_data import transform
import torch

trainSet = ChestXRay14(transform=transform)
trainSize = int(0.7 * len(trainSet))
valSize = len(trainSet) - trainSize

trainSet, valSet = torch.utils.data.random_split(trainSet, [trainSize, valSize])

trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
valLoader = DataLoader(valSet, batch_size=32)

#TODO: Load model here
model = ""

#Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Tune this number
epochs = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_idx, sample in enumerate(trainLoader):
        data = sample['image'].to(device)
        target = sample['label'].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in valLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

    print(
        f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(trainLoader):.4f}, '
        f'Val Loss: {val_loss / len(valLoader):.4f}')

    # Save the model
torch.save(model.state_dict(), 'pigaze_model5.pth')
