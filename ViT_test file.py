import numpy as np
import torch
import torch.nn as nn
import deeplake
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
    multilabel_confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = [sample[key] for sample in batch]
    return collated

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

def load_test_data(batch_size=32):
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
    test_loader = ds_test.pytorch(
        batch_size=batch_size,
        shuffle=False,
        decode_method={"images": "pil"},
        collate_fn=custom_collate_fn
    )
    return test_loader, ds_test

def test(model, test_loader, transform_pipeline, device):
    model.eval()
    model.to(device)

    ground_truth = []
    predictions = []
    prediction_confidences = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            pil_images = batch.get("images")
            processed_images = [transform_pipeline(img) for img in pil_images]
            images = torch.stack(processed_images).to(device)

            raw_labels = batch.get("labels") or batch.get("findings")
            labels = convert_labels_to_multihot(raw_labels, num_classes=15).to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            ground_truth.extend(labels.detach().cpu().numpy())
            predictions.extend(preds.detach().cpu().numpy())
            prediction_confidences.extend(probs.mean(dim=1).detach().cpu().numpy())

    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    f1_micro = f1_score(ground_truth, predictions, average='micro')
    f1_macro = f1_score(ground_truth, predictions, average='macro')
    precision = precision_score(ground_truth, predictions, average='micro')
    recall = recall_score(ground_truth, predictions, average='micro')
    hamming = hamming_loss(ground_truth, predictions)

    try:
        auc = roc_auc_score(ground_truth, predictions, average='macro')
    except:
        auc = None

    print("\n=============================================")
    print(f"Testing Finished")
    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall (micro): {recall:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    if auc is not None:
        print(f"AUC Score (macro): {auc:.4f}")
    else:
        print("AUC Score could not be computed")
    print("=============================================\n")

    cm = multilabel_confusion_matrix(ground_truth, predictions)
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    axs = axs.ravel()

    for i in range(15):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i])
        disp.plot(ax=axs[i], values_format='d')
        axs[i].set_title(f"Label {i}")
        axs[i].set_xlabel('Pred')
        axs[i].set_ylabel('True')

    plt.tight_layout()
    plt.savefig("Confusion_Matrices.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.hist(prediction_confidences, bins=50)
    plt.title("Average Prediction Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.savefig("Prediction_Confidence.png")
    plt.close()

def test_live_examples(model, ds_test, transform_pipeline, device, num_samples=5):
    model.eval()
    model.to(device)

    print("\n================ Live Predictions ================\n")
    for i in range(num_samples):
        img = ds_test[i]["images"].numpy()
        label = ds_test[i]["labels"].numpy() if "labels" in ds_test[i] else ds_test[i]["findings"].numpy()
        label = [int(l) for l in label]

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype != np.uint8:
            img = (255 * img).astype(np.uint8)

        img_pil = Image.fromarray(img)
        processed = transform_pipeline(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(processed)
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze().cpu().numpy()

        predicted_labels = [i for i, val in enumerate(preds) if val == 1]
        print(f"Example {i+1}")
        print(f"True Labels     : {label}")
        print(f"Predicted Labels: {predicted_labels}")
        print()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model and data for testing...")
    model = ViTModelWrapper(num_classes=15)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    test_loader, ds_test = load_test_data(batch_size=32)
    transform_pipeline = get_transforms()
    test(model, test_loader, transform_pipeline, device)
    test_live_examples(model, ds_test, transform_pipeline, device, num_samples=15)

if __name__ == "__main__":
    main()

"""
=============================================
Testing Finished
Precision (micro): 0.5142
Recall (micro): 0.2759
F1 Score (micro): 0.3591
F1 Score (macro): 0.1794
Hamming Loss: 0.0951
AUC Score (macro): 0.5524
=============================================
"""