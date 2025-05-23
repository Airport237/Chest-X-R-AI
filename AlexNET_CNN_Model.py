"""
This single Python file combines all the tasks required for our project:
    1. Data loading from DeepLake.
    2. Preprocessing of images.
    3. CNN model creation.
    4. Training loop with metric calculations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import deeplake
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

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
        batch_size=32,
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

# Model Creation
class AlexNetCNN(nn.Module):
    def __init__(self, num_classes=15):
        super(AlexNetCNN, self).__init__()
        print("Building CNN Model")
        # The feature extracts image features using convolution, batch normalization
        # ReLU activations, and pooling to reduce dimensions

        self.layer1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.layer2 = nn.Conv2d(96, 256, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Reduces dimensions by a factor of 2

            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(2)
        )
        """
        The classifier runs FFNN layers to further learn features about the predictions
        """

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            # Outputs raw logits for each class.
            nn.Linear(1000, num_classes)

        )
        print("CNN model built successfully!")
        print("Model Architecture:")
        print(self)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # x = self.layer1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.layer2(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.layer3(x)
        # x = self.relu(x)
        # x = self.layer4(x)
        # x = self.relu(x)
        # x = self.layer5(x)
        # x = self.relu(x)
        x = self.features(x)

        #if x.requires_grad:
        h = x.register_hook(self.activations_hook)

        x = self.maxpool(x)

        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features(x)


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
def train_model(model, train_loader, transform_pipeline, device, num_epochs=5, saved_model = ""):
    if (saved_model != ""):
        model.load_state_dict(torch.load(saved_model))
    epoch_losses = []
    epochs = []
    print("Starting Training Loop........")
    model.to(device)
    print(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
            labels = convert_labels_to_multihot(raw_labels, num_classes=15).float().to(device)

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
        plt.savefig("AlexNet Loss Graph")

        print("Saving model as 'AlexInter.pth'")
        torch.save(model.state_dict(), "AlexInter.pth")

        # Calculating metrics
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
        print("=============================================\n")

    print("========== Training Loop Finished ==========")
    print("Saving model as 'model.pth'")
    torch.save(model.state_dict(), "Alex.pth")
    print("Loss values: " + str(epoch_losses))


def test(model, test_loader, transform_pipeline, device, saved_model = ""):
    if (saved_model != ""):
        model.load_state_dict(torch.load(saved_model))
    model.eval()
    print("Starting Testing Loop")
    model.to(device)
    print(device)

    ground_truth = []
    predictions = []

    #Trying enabled gradient for GRAD-CAM
    with torch.set_grad_enabled(True):

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

    #accuracy = accuracy_score(ground_truth, predictions)
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
    print(f"AUC: {auc:.4f}")
    #print(f"Accuracy: {accuracy:.4f}")
    print("=============================================\n")

def visualize(model, test_loader, transform_pipeline, saved_model):
    if (saved_model != ""):
        model.load_state_dict(torch.load(saved_model))
    model.eval()
    # Get the first conv layer


    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.layer1.register_forward_hook(get_activation("features"))
    for batch in test_loader:
        orig_im = batch.get("images")[0]
        images = batch.get("images")[0]
        processed_images = [transform_pipeline(images)]
        images = torch.stack(processed_images)
        break
    output = model(images)
    act = activation["layer1"]
    act2 = activation["layer5"]

    fig, axarr = plt.subplots(4, 4, figsize=(12, 12))

    for idx in range(16):  # Visualize first 16 feature maps
        ax = axarr[idx // 4, idx % 4]
        ax.imshow(act[0][idx].cpu(), cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    fig3, axarr3 = plt.subplots(4, 4, figsize=(12, 12))

    for idx in range(16):  # Visualize first 16 feature maps
        ax = axarr3[idx // 4, idx % 4]
        ax.imshow(act2[0][idx].cpu(), cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.imshow(orig_im, cmap='viridis')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def grad_cam(model, test_loader, transform_pipeline, device, saved_model):
    if (saved_model != ""):
        model.load_state_dict(torch.load(saved_model))
    model.eval()

    #Track Grads for GRAD-CAM
    with torch.set_grad_enabled(True):

        for batch in test_loader:
            orig_im = batch.get("images")[14]
            images = batch.get("images")[14]
            processed_images = [transform_pipeline(images)]
            images = torch.stack(processed_images)
            images.requires_grad = True
            break

        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        te = outputs.detach().cpu().numpy()
        predicted = (outputs > 0.25).float()
        model.zero_grad()
        #Choose the class you want to analyze
        outputs[0, predicted].backward()

        gradients = model.get_activations_gradient()

        pooled_gradients = torch.mean(gradients, dim=(0, 2, 3))

        activations = model.get_activations(images).detach()

        for i in range(255):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        plt.matshow(heatmap.squeeze())

        #Image size hard coded as 227 x 227 for now
        heatmap = cv2.resize(heatmap.detach().cpu().numpy(), (227, 227))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        #Justi grabs one imgage for now
        img = images.detach().cpu().numpy()[0]
        # convert to [H, W, C]
        img = np.transpose(img, (1, 2, 0))
        # scale and convert to uint8
        img = (img * 255).astype(np.uint8)

        superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)
        cv2.imwrite('./map.jpg', superimposed_img)
        cv2.imwrite('./heatmap.jpg', heatmap)
        cv2.imwrite('./img.jpg', orig_im)



def main():
    print("========== Starting Model Training Process ==========")
    train_loader, test_loader, ds_train, ds_test = custom_get_data()
    transform_pipeline = get_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AlexNetCNN(num_classes=15)
    train_model(model, train_loader, transform_pipeline, device, num_epochs=20)
    print("========== Model Training is Complete ==========")
    test(model, test_loader, transform_pipeline, device, saved_model = "Alex.pth")
    #visualize(model, test_loader, transform_pipeline)
    #grad_cam(model, test_loader, transform_pipeline, device,"Alex.pth")



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



"""
10 Epoch LR 0.01 Batch 64 Current Model pth for resnet.pth
=============================================
Testing Finished
Precision (micro): 0.5613
Recall (micro): 0.1513
F1 Score (micro): 0.2384
F1 Score (macro): 0.0638
Hamming Loss: 0.0927
AUC: 0.5170
=============================================
"""