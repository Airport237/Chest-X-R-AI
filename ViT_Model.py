"""
This single Python file combines all the tasks required for our project:
    1. Data loading from DeepLake.
    2. Preprocessing of images.
    3. Vision Transformer model creation.
    4. Training and Testing loop with metric calculations and AUC.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import deeplake
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score, roc_auc_score
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import time


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated


def custom_get_data(batch_size=32):
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
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
    return train_loader, test_loader, ds_train, ds_test


def get_transforms():
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    def transform_fn(pil_img):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return processor(images=pil_img, return_tensors="pt")["pixel_values"].squeeze(0)
    return transform_fn


class ViTModelWrapper(nn.Module):
    def __init__(self, num_classes=15):
        super(ViTModelWrapper, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    def forward(self, x):
        return self.model(x).logits


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


def train_model(model, train_loader, transform_pipeline, device, num_epochs=5):
    epoch_losses = []
    epoch_accuracies = []
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        epoch_preds = []
        epoch_targets = []

        for batch in train_loader:
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
            preds = (torch.sigmoid(outputs) > 0.5).float()
            epoch_preds.append(preds.detach().cpu())
            epoch_targets.append(labels.detach().cpu())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        epoch_preds = torch.cat(epoch_preds, dim=0).numpy()
        epoch_targets = torch.cat(epoch_targets, dim=0).numpy()

        try:
            auc = roc_auc_score(epoch_targets, epoch_preds, average='macro')
            print(f"AUC Score (macro): {auc:.4f}")
        except:
            auc = float('nan')

        accuracy = accuracy_score(epoch_targets, epoch_preds)
        epoch_accuracies.append(accuracy)
        f1_micro = f1_score(epoch_targets, epoch_preds, average='micro')
        f1_macro = f1_score(epoch_targets, epoch_preds, average='macro')
        precision = precision_score(epoch_targets, epoch_preds, average='micro')
        recall = recall_score(epoch_targets, epoch_preds, average='micro')
        hamming = hamming_loss(epoch_targets, epoch_preds)

        print(f"\nEpoch {epoch + 1}/{num_epochs} finished in {time.time() - start_time:.2f} seconds")
        print(f"Average Loss: {epoch_loss:.4f}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Micro: {f1_micro:.4f}\nF1 Macro: {f1_macro:.4f}\nHamming Loss: {hamming:.4f}")

    plt.figure()
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label='Loss')
    plt.plot(range(1, len(epoch_accuracies)+1), epoch_accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss and Accuracy over Epochs')
    plt.legend()
    plt.savefig("training_metrics.png")
    plt.close()
    torch.save(model.state_dict(), "model.pth")

"""
def test(model, test_loader, transform_pipeline, device, saved_model="model.pth"):
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    model.to(device)
    ground_truth = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            pil_images = batch.get("images")
            processed_images = [transform_pipeline(img) for img in pil_images]
            images = torch.stack(processed_images).to(device)
            raw_labels = batch.get("labels") or batch.get("findings")
            labels = convert_labels_to_multihot(raw_labels, num_classes=15).to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()
            ground_truth.append(labels.cpu().numpy())
            predictions.append(predicted.cpu().numpy())

    ground_truth = np.concatenate(ground_truth, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    try:
        auc = roc_auc_score(ground_truth, predictions, average='macro')
        print(f"AUC Score (macro): {auc:.4f}")
    except:
        print("AUC Score could not be computed.")

    accuracy = accuracy_score(ground_truth, predictions)
    f1_micro = f1_score(ground_truth, predictions, average='micro')
    f1_macro = f1_score(ground_truth, predictions, average='macro')
    precision = precision_score(ground_truth, predictions, average='micro')
    recall = recall_score(ground_truth, predictions, average='micro')
    hamming = hamming_loss(ground_truth, predictions)

    print(f"\nTesting Finished\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Micro: {f1_micro:.4f}\nF1 Macro: {f1_macro:.4f}\nHamming Loss: {hamming:.4f}")

    plt.figure()
    plt.plot(predictions.mean(axis=1), label='Prediction Confidence')
    plt.title('Average Prediction Confidence')
    plt.savefig("test_prediction_confidence.png")
    plt.close()

"""
def main():
    train_loader, test_loader, ds_train, ds_test = custom_get_data(batch_size=32)
    transform_pipeline = get_transforms()
    device = torch.device("cpu")
    model = ViTModelWrapper(num_classes=15)
    train_model(model, train_loader, transform_pipeline, device, num_epochs=5)
    # test(model, test_loader, transform_pipeline, device)



if __name__ == "__main__":
    main()
"""
AUC Score (macro): 0.5166

Epoch 1/5 finished in 11790.19 seconds
Average Loss: 0.1834
Accuracy: 0.4820
Precision: 0.6935
Recall: 0.4086
F1 Micro: 0.5142
F1 Macro: 0.0707
Hamming Loss: 0.0621
AUC Score (macro): 0.5310

Epoch 2/5 finished in 11900.85 seconds
Average Loss: 0.1639
Accuracy: 0.4945
Precision: 0.7196
Recall: 0.4347
F1 Micro: 0.5420
F1 Macro: 0.1085
Hamming Loss: 0.0591
AUC Score (macro): 0.5467

Epoch 3/5 finished in 11908.96 seconds
Average Loss: 0.1558
Accuracy: 0.5099
Precision: 0.7293
Recall: 0.4604
F1 Micro: 0.5645
F1 Macro: 0.1539
Hamming Loss: 0.0572
AUC Score (macro): 0.5707

Epoch 4/5 finished in 11838.62 seconds
Average Loss: 0.1460
Accuracy: 0.5350
Precision: 0.7463
Recall: 0.5003
F1 Micro: 0.5990
F1 Macro: 0.2162
Hamming Loss: 0.0539
AUC Score (macro): 0.6061

Epoch 5/5 finished in 11876.27 seconds
Average Loss: 0.1312
Accuracy: 0.5712
Precision: 0.7745
Recall: 0.5569
F1 Micro: 0.6480
F1 Macro: 0.2978
Hamming Loss: 0.0487
AUC Score (macro): 0.5524

Testing Finished
Accuracy: 0.2667
Precision: 0.5142
Recall: 0.2759
F1 Micro: 0.3591
F1 Macro: 0.1794
Hamming Loss: 0.0951

Process finished with exit code 0
"""