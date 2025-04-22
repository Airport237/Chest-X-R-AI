# Vision Transformer (ViT) for ChestX-ray14 - Final Version with Explicit Class 0 Dropping

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # macOS SSL fix

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import deeplake
import matplotlib.pyplot as plt
import random
import warnings

from torchvision import transforms, models
from torchvision.models.vision_transformer import ViT_B_16_Weights
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, roc_auc_score, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ================= Custom Collate ================= #
def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated


# ========== Data Analysis Helper ========== #
def print_class_distribution(ds, name="Dataset"):
    label_tensor = 'labels' if 'labels' in ds.tensors else 'findings'
    class_counts = [0] * 15

    for i, sample in enumerate(ds):
        labels = sample[label_tensor].numpy()
        indices = np.where(labels == 1)[0]
        for idx in indices:
            class_counts[idx] += 1

    print(f"Class Distribution in {name}: ")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}: {count} samples")

        # ========== Explicit Class 0 Drop Function ========== #


def drop_class_0_samples(ds, drop_count):
    label_tensor = 'labels' if 'labels' in ds.tensors else 'findings'
    class_0_indices = []
    other_indices = []

    print("Scanning dataset for class 0 samples...")
    for i, sample in enumerate(ds):
        labels = sample[label_tensor].numpy()
        if 0 in np.where(labels == 1)[0]:
            class_0_indices.append(i)
        else:
            other_indices.append(i)

    print(f"Total class 0 samples found: {len(class_0_indices)}")
    print(f"Dropping {drop_count} class 0 samples...")

    if drop_count > len(class_0_indices):
        raise ValueError(f"Only {len(class_0_indices)} class 0 samples available to drop, but requested {drop_count}")

    retained_class_0_indices = class_0_indices[drop_count:]
    final_indices = retained_class_0_indices + other_indices
    print(f"Final dataset size after drop: {len(final_indices)}")
    return ds[final_indices]


# ========== Load & Filter Dataset ========== #
def custom_get_data_filtered():
    print("Loading datasets from DeepLake...")
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')

    print("Analyzing TRAIN dataset before dropping...")
    print_class_distribution(ds_train, "Train Set")

    print("Analyzing TEST dataset before dropping...")
    print_class_distribution(ds_test, "Test Set")
    print("Loading datasets from DeepLake...")
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')

    print("Dropping 45000 samples from class 0 in training set...")
    ds_train_filtered = drop_class_0_samples(ds_train, drop_count=45000)

    print("Dropping 5000 samples from class 0 in test set...")
    ds_test_filtered = drop_class_0_samples(ds_test, drop_count=5000)

    train_loader = ds_train_filtered.pytorch(batch_size=32, num_workers=0, shuffle=True,
                                             decode_method={"images": "pil"}, collate_fn=custom_collate_fn)
    test_loader = ds_test_filtered.pytorch(batch_size=32, shuffle=False,
                                           decode_method={"images": "pil"}, collate_fn=custom_collate_fn)

    print("Data loaders created successfully.")
    return train_loader, test_loader, ds_test_filtered


# ========== Transforms for ViT ========== #
def get_vit_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


# ========== Vision Transformer Model ========== #
class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=weights)
        in_features = self.vit.heads[0].in_features
        self.vit.heads = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.vit(x)


# ========== Convert Multihot to Binary Label ========== #
def convert_to_one_vs_all_labels(raw_labels, class_idx):
    binary_labels = []
    for label_list in raw_labels:
        label_tensor = label_list if isinstance(label_list, list) else label_list.tolist()
        match = any(int(item) == class_idx for item in label_tensor)
        binary_labels.append(torch.tensor([1.0 if match else 0.0]))
    return torch.stack(binary_labels)


# ========== Training Loop ========== #
def train_model(model, train_loader, transform, device, class_idx, epochs=15):
    model.to(device)
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            images = torch.stack([transform(img) for img in batch['images']]).to(device)
            label_key = 'labels' if 'labels' in batch else 'findings'
            labels = convert_to_one_vs_all_labels(batch[label_key], class_idx).to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        epoch_loss = total_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}")
    return train_losses


# ========== Evaluation Loop ========== #
def evaluate_model(model, test_loader, transform, device, class_idx):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    test_losses = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in test_loader:
            images = torch.stack([transform(img) for img in batch['images']]).to(device)
            label_key = 'labels' if 'labels' in batch else 'findings'
            labels = convert_to_one_vs_all_labels(batch[label_key], class_idx).to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            loss = criterion(outputs, labels.squeeze())
            test_losses.append(loss.item() * labels.size(0))
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)
    avg_loss = np.sum(test_losses) / len(test_loader.dataset)

    pos_label = 1 if class_idx != 0 else 0

    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = float('nan')

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}")
    print(f"Hamming Loss: {hamming:.4f}, AUC Score: {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    return avg_loss, precision, recall, f1_micro, f1_macro, hamming, auc


# ========== Run for All Classes ========== #
def run_one_vs_all(classes=range(15), epochs=15):
    train_loader, test_loader, ds_test = custom_get_data_filtered()
    transform = get_vit_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for class_idx in classes:
        print(f"\n=== Training Class {class_idx} vs All ===")
        model = ViTBinaryClassifier()
        train_losses = train_model(model, train_loader, transform, device, class_idx, epochs)
        test_loss, precision, recall, f1_micro, f1_macro, hamming, auc = evaluate_model(
            model, test_loader, transform, device, class_idx
        )

        model_path = f"vit_class_{class_idx}.pth"
        torch.save(model.state_dict(), model_path)
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.axhline(y=test_loss, color='red', linestyle='--', label='Test Loss')
        plt.title(f"Loss Curve for Class {class_idx}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_curve_class_{class_idx}.png")
        plt.close()

        # Live Testing on Random Samples
        print(f"\nLive Testing for Class {class_idx}:")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to(device)
        for i in random.sample(range(len(ds_test)), 5):
            img = ds_test[i]['images'].numpy()
            img = transforms.ToPILImage()(img)
            label_raw = ds_test[i]['labels'] if 'labels' in ds_test.tensors else ds_test[i]['findings']
            label_val = 1.0 if class_idx in label_raw else 0.0
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x).squeeze()
                prob = torch.sigmoid(out).item()
                pred = 1 if prob > 0.5 else 0
            print(f"  Sample #{i}: Predicted={pred} | True={int(label_val)} | Confidence={prob:.3f}")


# ========== Start Training ========== #
run_one_vs_all(classes=range(15), epochs=15)
