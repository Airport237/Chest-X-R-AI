import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import deeplake
import matplotlib.pyplot as plt

from torchvision import transforms, models
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, roc_auc_score


# Collate Function
def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated


# Filter Class 0
def filter_class_0_samples(ds, keep_ratio=0.2, max_samples=50000):
    class0_indices, other_indices = [], []
    for i in range(min(len(ds), max_samples)):
        labels = ds[i]['labels'].numpy()
        label_indices = np.where(labels == 1)[0]
        if 0 in label_indices:
            if np.random.rand() < keep_ratio:
                class0_indices.append(i)
        elif len(label_indices) > 0:
            other_indices.append(i)
    return class0_indices + other_indices


# Data Loader with Filtering
def custom_get_data_filtered(keep_ratio=0.2):
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
    train_keep = filter_class_0_samples(ds_train, keep_ratio=keep_ratio)
    test_keep = filter_class_0_samples(ds_test, keep_ratio=0.5)
    ds_train_filtered = ds_train[train_keep]
    ds_test_filtered = ds_test[test_keep]
    train_loader = ds_train_filtered.pytorch(batch_size=32, num_workers=2, shuffle=True,
                                             decode_method={"images": "pil"}, collate_fn=custom_collate_fn)
    test_loader = ds_test_filtered.pytorch(batch_size=32, shuffle=False,
                                           decode_method={"images": "pil"}, collate_fn=custom_collate_fn)
    return train_loader, test_loader


# ViT Preprocessing
def get_vit_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


# ViT Binary Classifier
class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Linear(self.vit.heads.head.in_features, 1)

    def forward(self, x):
        return self.vit(x)


# Label Conversion (one-vs-all)
def convert_to_one_vs_all_labels(raw_labels, class_idx):
    binary_labels = []
    for label_list in raw_labels:
        match = any(int(item) == class_idx for item in label_list)
        binary_labels.append(torch.tensor([1.0 if match else 0.0]))
    return torch.stack(binary_labels)


# Training Loop
def train_model(model, train_loader, transform, device, class_idx, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            images = torch.stack([transform(img) for img in batch['images']]).to(device)
            labels = convert_to_one_vs_all_labels(batch['labels'], class_idx).to(device)
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


# Evaluation
def evaluate_model(model, test_loader, transform, device, class_idx):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    test_losses = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in test_loader:
            images = torch.stack([transform(img) for img in batch['images']]).to(device)
            labels = convert_to_one_vs_all_labels(batch['labels'], class_idx).to(device)
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

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    avg_loss = np.sum(test_losses) / len(test_loader.dataset)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}")
    print(f"Hamming Loss: {hamming:.4f}, AUC Score: {auc:.4f}")

    return avg_loss, precision, recall, f1_micro, f1_macro, hamming, auc


# Run One-vs-All for Each Class
def run_one_vs_all(classes=range(15), epochs=5):
    train_loader, test_loader = custom_get_data_filtered()
    transform = get_vit_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for class_idx in classes:
        print(f"\n=== Training Class {class_idx} vs All ===")
        model = ViTBinaryClassifier()
        train_losses = train_model(model, train_loader, transform, device, class_idx, epochs)
        test_loss, precision, recall, f1_micro, f1_macro, hamming, auc = evaluate_model(
            model, test_loader, transform, device, class_idx
        )

        # Save model and plot
        torch.save(model.state_dict(), f"vit_class_{class_idx}.pth")

        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.axhline(y=test_loss, color='red', linestyle='--', label='Test Loss')
        plt.title(f"Loss Curve for Class {class_idx}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_curve_class_{class_idx}.png")
        plt.close()


# Run it
run_one_vs_all(classes=range(15), epochs=15)