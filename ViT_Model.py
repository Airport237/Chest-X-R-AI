"""
This single Python file combines all the tasks required for our project:
    1. Data loading from DeepLake.
    2. Preprocessing of images.
    3. Vision Transformer model creation.
    4. Training loop with metric calculations and AUC.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import deeplake
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score, roc_auc_score, roc_curve
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Custom Collate Function
def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated

# Data Loading Function
def custom_get_data(batch_size=32):
    print("Data Loading......")
    print("Loading training dataset from DeepLake...")
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    print("Training dataset loaded successfully!")

    print("Loading test dataset from DeepLake...")
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
    print("Test dataset loaded successfully!")

    print(f"Total training samples: {len(ds_train)}")
    print(f"Total test samples: {len(ds_test)}")

    print("Creating DataLoaders with custom collate function...")
    train_loader = ds_train.pytorch(
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
        decode_method={"images": "pil"},
        collate_fn=custom_collate_fn
    )

    test_loader = ds_test.pytorch(
        batch_size=batch_size,
        shuffle=False,
        decode_method={"images": "pil"},
        collate_fn=custom_collate_fn
    )

    print("DataLoaders created successfully!")
    return train_loader, test_loader, ds_train, ds_test

# Preprocessing
def get_transforms():
    print("Preprocessing Pipeline........")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def transform_fn(pil_img):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)

    print("Preprocessing pipeline is ready.")
    return transform_fn

# Model Creation
class ViTModelWrapper(nn.Module):
    def __init__(self, num_classes=15):
        super(ViTModelWrapper, self).__init__()
        print("Building Vision Transformer Model...")
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        print("ViT model built successfully!")
        print("Model Architecture:")
        print(self.model)

    def forward(self, x):
        return self.model(x).logits

# Label Conversion Helper
def convert_labels_to_multihot(raw_labels, num_classes=15):
    processed_labels = []
    for label in raw_labels:
        multi_hot = torch.zeros(num_classes, dtype=torch.float)
        for item in label:
            try:
                idx = int(item)
                if 0 <= idx < num_classes:
                    multi_hot[idx] = 1.0
            except Exception:
                continue
        processed_labels.append(multi_hot)
    return torch.stack(processed_labels)

# Training Loop
def train_model(model, train_loader, transform_pipeline, device, num_epochs=3):
    epoch_losses = []
    print("Starting Training Loop........")
    model.to(device)
    print(f"Training on device: {device}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()

    for epoch in range(num_epochs):
        print("\n---------------------------------------------")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        batch_count = 0
        epoch_preds = []
        epoch_targets = []

        for batch in train_loader:
            batch_count += 1
            print(f"\nProcessing batch {batch_count}...")

            pil_images = batch.get("images")
            processed_images = [transform_pipeline(img) for img in pil_images]
            images = torch.stack(processed_images).to(device)

            raw_labels = batch.get("labels") or batch.get("findings")
            labels = convert_labels_to_multihot(raw_labels, num_classes=15).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            print(f"  Batch {batch_count} Loss: {loss.item():.4f}")

            preds = (torch.sigmoid(outputs) > 0.5).float()
            epoch_preds.append(preds.detach().cpu())
            epoch_targets.append(labels.detach().cpu())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        epoch_preds = torch.cat(epoch_preds, dim=0).numpy()
        epoch_targets = torch.cat(epoch_targets, dim=0).numpy()

        plt.figure()
        plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.savefig("Loss Graph")
        plt.close()

        try:
            auc = roc_auc_score(epoch_targets, epoch_preds, average='macro')
            print(f"AUC Score (macro): {auc:.4f}")
        except:
            print("AUC Score could not be computed for this epoch.")

        accuracy = accuracy_score(epoch_targets, epoch_preds)
        f1_micro = f1_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        f1_macro = f1_score(epoch_targets, epoch_preds, average='macro', zero_division=0)
        precision = precision_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        recall = recall_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        hamming = hamming_loss(epoch_targets, epoch_preds)

        print("\n=============================================")
        print(f"Epoch {epoch + 1} completed.")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Precision (micro): {precision:.4f}")
        print(f"Recall (micro): {recall:.4f}")
        print(f"F1 Score (micro): {f1_micro:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print("=============================================\n")

    print("========== Training Loop Finished ==========")
    print("Saving model as 'model.pth'")
    torch.save(model.state_dict(), "model.pth")

# Main function
def main():
    print("========== Starting Model Training Process ==========")
    train_loader, test_loader, ds_train, ds_test = custom_get_data(batch_size=32)
    transform_pipeline = get_transforms()
    device = torch.device("cpu")  # Using CPU to avoid SIGKILL issues
    print(f"Using device: {device}")
    model = ViTModelWrapper(num_classes=15)
    train_model(model, train_loader, transform_pipeline, device, num_epochs=3)
    print("========== Model Training is Complete ==========")

if __name__ == "__main__":
    main()
