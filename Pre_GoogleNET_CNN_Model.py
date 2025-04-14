"""
This single Python file combines all the tasks required for our project:
    1. Data loading from DeepLake.
    2. Preprocessing of images.
    3. CNN model creation.
    4. Training loop with metric calculations.
"""
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import deeplake
import matplotlib.pyplot as plt
from torchvision import transforms, models
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score, roc_auc_score


# Custom Collate Function

def custom_collate_fn(batch):
    """
    DeepLake's default collation method attempts to stack tensors, which causes errors when the
    samples vary in size. This function returns a dictionary where each key maps to a list of values
    from the batch, preventing the stacking error.
    """
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated

# Data Loading Function

def custom_get_data():
    print("Data Loading......")
    print("Loading training dataset from DeepLake...")
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    print("Training dataset loaded successfully!")

    print("Loading test dataset from DeepLake...")
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
    print("Test dataset loaded successfully!")

    print(f"Total training samples: {len(ds_train)}")
    print(f"Total test samples: {len(ds_test)}")

    print("Creating training DataLoader with custom collate function...")
    # Using our custom_collate_fn to prevent errors due to variable-sized labels
    train_loader = ds_train.pytorch(
        num_workers=2,
        batch_size=64,
        shuffle=True,
        decode_method={"images": "pil"},
        collate_fn=custom_collate_fn
    )

    test_loader = ds_test.pytorch(
        batch_size=64,
        shuffle=False,
        decode_method={"images": "pil"},
        collate_fn=custom_collate_fn
    )

    print("Testing DataLoader created successfully!")
    return train_loader, test_loader, ds_train, ds_test

# Preprocessing
def get_transforms():
    print("Preprocessing Pipeline........")
    # We first convert grayscale images to 3-channel images because our CNN model
    # expects 3-channel (RGB) input, then resize, convert to tensor, and normalize
    transform_pipeline = transforms.Compose([
        # Convert grayscale images to 3-channel images.
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    print("Preprocessing pipeline is ready.")
    return transform_pipeline

# # Model Creation
# class AlexNetCNN(nn.Module):
#     def __init__(self, num_classes=15):
#         super(AlexNetCNN, self).__init__()
#         print("Building CNN Model")
#
#         self.pre = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         self.fc = nn.Linear(self.fc.in_features, num_classes)
#
#         # The feature extracts image features using convolution, batch normalization
#         # ReLU activations, and pooling to reduce dimensions
#
#
#         print("CNN model built successfully!")
#         print("Model Architecture:")
#         print(self)
#
#     def forward(self, x):
#         x = self.pre(x)
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# Label Conversion Helper
def convert_labels_to_multihot(raw_labels, num_classes=15):
    """
    Many sample have labels of variable lengths, which causes errors when stacking,
    Tjis function converts each samples's labels into a fixed-length multi-hot vector.
    Each index in the vector corresponds to one of the 15 classes.
    """
    processed_labels = []
    for label in raw_labels:
        multi_hot = torch.zeros(num_classes, dtype=torch.float)
        # if isinstance(label, Int64):
        for item in label:
            try:
                idx = int(item)
            except Exception:
                continue
            if 0 <= idx < num_classes:
                multi_hot[idx] = 1.0
        # else:
        #     try:
        #         idx = int(label)
        #         if 0 <= idx < num_classes:
        #             multi_hot[idx] = 1.0
        #     except Exception:
        #         pass
        processed_labels.append(multi_hot)
    return torch.stack(processed_labels)

# Training Loop
def train_model(model, train_loader, transform_pipeline, device, num_epochs=5):
    epoch_losses = []
    epochs = []
    print("Starting Training Loop........")
    model.to(device)
    print(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(1, num_epochs):
        print("\n---------------------------------------------")
        print(f"Epoch {epoch}/{num_epochs}")
        epochs.append(epoch)
        running_loss = 0.0
        batch_count = 0
        epoch_preds = []
        epoch_targets = []

        # if (epoch % 10 == 0):
        #     for group in optimizer.param_groups:
        #         group['lr'] /= 10

        for batch in train_loader:
            batch_count += 1
            print(f"\nProcessing batch {batch_count}...")

            # Process images (assumes DataLoader returns a dict with key "images" as PIL images)
            pil_images = batch.get("images")
            if pil_images is None:
                raise KeyError("The batch does not contain the key 'images'. Check your data loader.")
            processed_images = [transform_pipeline(img) for img in pil_images]
            images = torch.stack(processed_images).to(device)

            # Process labels: check for "labels" or "findings"
            if "labels" in batch:
                raw_labels = batch["labels"]
            elif "findings" in batch:
                raw_labels = batch["findings"]
            else:
                raise KeyError("No label key found in batch (expected 'labels' or 'findings').")

            # Convert raw labels into fixed-length multi-hot vectors.
            labels = convert_labels_to_multihot(raw_labels, num_classes=15).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            print(f"  Batch {batch_count} Loss: {loss.item():.4f}")

            # Accumulate predictions and targets for metric calculation.
            preds = (torch.sigmoid(outputs) > 0.5).float()
            epoch_preds.append(preds.detach().cpu())
            epoch_targets.append(labels.detach().cpu())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        epoch_preds = torch.cat(epoch_preds, dim=0).numpy()
        epoch_targets = torch.cat(epoch_targets, dim=0).numpy()


        plt.plot(epoch_losses, label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("Google Loss Graph 10 Epoch")

        # Calculating metrics
        auc = roc_auc_score(epoch_targets, epoch_preds)

        f1_micro = f1_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        f1_macro = f1_score(epoch_targets, epoch_preds, average='macro', zero_division=0)
        precision = precision_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        recall = recall_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        hamming = hamming_loss(epoch_targets, epoch_preds)

        print("\n=============================================")
        print(f"Epoch {epoch} completed.")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Precision (micro): {precision:.4f}")
        print(f"Recall (micro): {recall:.4f}")
        print(f"F1 Score (micro): {f1_micro:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        print(f"AUC: {auc:.4f}")

        print("=============================================\n")

    print("========== Training Loop Finished ==========")
    print("Saving model as 'google.pth'")
    torch.save(model.state_dict(), "google.pth")


def test(model, test_loader, transform_pipeline, device, saved_model = ""):
    if (saved_model != ""):
        model.load_state_dict(torch.load(saved_model))
    model.eval()
    print("Starting Testing Loop")
    model.to(device)
    print(device)

    ground_truth = []
    predictions = []

    with torch.no_grad():

        for i, batch in enumerate(test_loader):
            print(f"Testing Batch #{i} of {len(test_loader)}")
            # Process images (assumes DataLoader returns a dict with key "images" as PIL images)
            pil_images = batch.get("images")
            if pil_images is None:
                raise KeyError("The batch does not contain the key 'images'. Check your data loader.")
            processed_images = [transform_pipeline(img) for img in pil_images]
            images = torch.stack(processed_images).to(device)

            # Process labels: check for "labels" or "findings"
            if "labels" in batch:
                raw_labels = batch["labels"]
            elif "findings" in batch:
                raw_labels = batch["findings"]
            else:
                raise KeyError("No label key found in batch (expected 'labels' or 'findings').")

            # Convert raw labels into fixed-length multi-hot vectors.
            labels = convert_labels_to_multihot(raw_labels, num_classes=15).to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()

            ground_truth.append(labels.detach().cpu().numpy()[0])
            predictions.append(predicted.detach().cpu().numpy()[0])


    # Calculating metrics

    accuracy = accuracy_score(ground_truth, predictions)
    f1_micro = f1_score(ground_truth, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    precision = precision_score(ground_truth, predictions, average='micro', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='micro', zero_division=0)
    hamming = hamming_loss(ground_truth, predictions)
    auc = roc_auc_score(ground_truth, predictions)


    print("\n=============================================")
    print(f"Testing Finished")
    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall (micro): {recall:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")

    print("=============================================\n")

def visualize(model):
    model.eval()
    # Get the first conv layer
    first_conv_layer = model.inception3a

    # Get the weights (filters) as a tensor
    filters = first_conv_layer.branch1.conv.weight.data.clone()

    min_val = filters.min()
    max_val = filters.max()
    filters = (filters - min_val) / (max_val - min_val)


    num = 6
    plt.figure(figsize=(num * 2, 2))
    for i in range(num):
        filter_img = filters[i].permute(1, 2, 0)  # CHW -> HWC
        plt.subplot(1, num, i + 1)
        plt.imshow(filter_img)
        plt.axis('off')
    plt.show()


def main():
    print("========== Starting Model Training Process ==========")
    start = datetime.now()
    print("Start time: ", start.strftime("%H:%M:%S"))
    train_loader, test_loader, ds_train, ds_test = custom_get_data()
    transform_pipeline = get_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #model = AlexNetCNN(num_classes=15)\
    num_classes = 15
    #Pretrained on IMAGENET1K_V1
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    train_model(model, train_loader, transform_pipeline, device, num_epochs=10)
    print("========== Model Training is Complete ==========")
    finish = datetime.now()

    # Print the time in HH:MM:SS format
    print("Finish time:", finish.strftime("%H:%M:%S"))
    print("Total Training time:", finish - start)
    test(model, test_loader, transform_pipeline, device, saved_model = "google.pth")
    #visualize(model)



if __name__ == "__main__":
    main()

##########################
# RESULTS
"""
=============================================
Epoch 5 completed.
Average Loss: 0.1989
Precision (micro): 0.5837
Recall (micro): 0.4801
F1 Score (micro): 0.5268
F1 Score (macro): 0.0490
Hamming Loss: 0.0694
Accuracy: 0.5796
=============================================
"""
##########################

"""
=============================================
Testing Finished
Precision (micro): 0.3550
Recall (micro): 0.2459
F1 Score (micro): 0.2905
F1 Score (macro): 0.0349
Hamming Loss: 0.1156
Accuracy: 0.3550
=============================================

Accuracy: 35.500
"""