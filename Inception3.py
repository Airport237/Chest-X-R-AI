import torch
import torch.nn as nn
import torch.optim as optim
import deeplake
from matplotlib import pyplot as plt
from torchvision import models
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score, roc_auc_score
import numpy as np

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
        num_workers=4,
        batch_size=64,
        shuffle=True,
        decode_method={"images": "pil"},
        collate_fn=custom_collate_fn
    )

    test_loader = ds_test.pytorch(
        batch_size=96,
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
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])
    ])
    print("Preprocessing pipeline is ready.")
    return transform_pipeline


# Label Conversion Helper
def convert_labels_to_multihot(raw_labels, num_classes=14):
    """
    Many sample have labels of variable lengths, which causes errors when stacking,
    Tjis function converts each samples's labels into a fixed-length multi-hot vector.
    Each index in the vector corresponds to one of the 14 classes.
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
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
    model.train()

    for epoch in range(num_epochs):
        print("\n---------------------------------------------")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epochs.append(epoch + 1)
        running_loss = 0.0
        batch_count = 0
        epoch_preds = []
        epoch_targets = []

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
            outputs, aux_outputs = model(images)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + .4 * loss2
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
        plt.savefig("Loss Graph")

        # Calculating metrics
        accuracy = accuracy_score(epoch_targets, epoch_preds)
        f1_micro = f1_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        f1_macro = f1_score(epoch_targets, epoch_preds, average='macro', zero_division=0)
        precision = precision_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        recall = recall_score(epoch_targets, epoch_preds, average='micro', zero_division=0)
        hamming = hamming_loss(epoch_targets, epoch_preds)
        auc = roc_auc_score(epoch_targets, epoch_preds)
        print("\n=============================================")


        print(f"Recall (micro): {recall:.4f}")
        print(f"F1 Score (micro): {f1_micro:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f'AUC: {auc:.4f}')
        print("=============================================\n")

    print("========== Training Loop Finished ==========")
    print("Saving model as 'model.pth'")
    torch.save(model.state_dict(), "model.pth")


def test(model, test_loader, transform_pipeline, device, saved_model = ""):
    if (saved_model != ""):
        model.load_state_dict(torch.load(saved_model))
        model.eval()
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

            ground_truth.extend(labels.detach().cpu().numpy())
            predictions.extend(predicted.detach().cpu().numpy())

    # Calculating metrics

    accuracy = accuracy_score(ground_truth, predictions)
    f1_micro = f1_score(ground_truth, predictions, average='micro', zero_division=0)
    f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    precision = precision_score(ground_truth, predictions, average='micro', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='micro', zero_division=0)
    hamming = hamming_loss(ground_truth, predictions)
    auc = roc_auc_score(ground_truth, predictions, average='macro')


    print("\n=============================================")
    print(f"Testing Finished")
    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall (micro): {recall:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Area Under Curve (AUC): {auc:.4f}")
    print("=============================================\n")

def main():
    print("========== Starting Model Training Process ==========")
    train_loader, test_loader, ds_train, ds_test = custom_get_data()
    transform_pipeline = get_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_classes = 15
    model = models.inception_v3(pretrained=True, aux_logits=True)
    freeze = True
    for name, param in model.named_parameters():
        if "Mixed_7a" in name:
            freeze = False  # start unfreezing here
        param.requires_grad = not freeze


    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    print(model)
    train_model(model, train_loader, transform_pipeline, device, num_epochs=20)

    print("========== Model Training is Complete ==========")
    test(model, test_loader, transform_pipeline, device, saved_model = "model.pth")

if __name__ == "__main__":
    main()