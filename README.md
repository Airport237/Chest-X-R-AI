# Chest-X-R-AI

Multi-label classification of thoracic pathologies on the NIH Chest X-ray dataset. The repository compares several deep-learning architectures — a custom baseline CNN, AlexNet, VGG16, ResNet50 (from-scratch and pretrained), Inception V3, and a Vision Transformer — on the same 15-class problem with a shared data and evaluation pipeline.

## Dataset

Images and labels are streamed from DeepLake:

- `hub://activeloop/nih-chest-xray-train`
- `hub://activeloop/nih-chest-xray-test`

Each X-ray can have multiple findings, so labels are encoded as fixed-length 15-dim multi-hot vectors. Models are trained with `BCEWithLogitsLoss` and predictions are produced by thresholding `sigmoid(logits) > 0.5`.

## Models

| File | Architecture | Notes |
| --- | --- | --- |
| `Baseline_CNN_Model.py` | Custom 3-block CNN | Simple from-scratch baseline |
| `AlexNET_CNN_Model.py` | AlexNet-style CNN | From scratch |
| `VGG16_CNN_Model.py` | `torchvision.models.vgg16` | ImageNet pretrained, classifier head replaced |
| `ResNET_CNN_Model.py` | `torchvision.models.resnet50` | Trained from scratch, includes Grad-CAM visualization |
| `Pre_ResNET_CNN_Model.py` | ResNet50 (`ResNet50_Weights.DEFAULT`) | Transfer learning + Grad-CAM heatmaps |
| `Inception3.py` | Inception V3 | Pretrained, partial unfreezing from `Mixed_7a` onward |
| `Vision_Transformer_model.ipynb` | `google/vit-base-patch16-224-in21k` | Hugging Face ViT, fine-tuned with confusion matrices and ROC curves |

## Setup

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins CUDA 11.8 builds of PyTorch (`torch==2.6.0+cu118`, `torchvision==0.21.0+cu118`). On a CPU-only machine or a different CUDA version, install matching wheels from <https://pytorch.org> instead.

## Running a model

Each `*.py` file is self-contained — it loads the data, builds the model, trains, and tests:

```bash
python Baseline_CNN_Model.py
python VGG16_CNN_Model.py
python Pre_ResNET_CNN_Model.py
# etc.
```

For the ViT, open `Vision_Transformer_model.ipynb` in Jupyter and run the cells top-to-bottom.

### Hyperparameters

Adjust per file:

- **Batch size** — in `custom_get_data()` where `train_loader` / `test_loader` are built.
- **Learning rate** — inside `train_model()` where the Adam optimizer is constructed.
- **Epochs** — `num_epochs` argument passed to `train_model()` from `main()`.

Trained weights are saved to `model.pth` (or a model-specific name like `resnet.pth`, `ViT_model.pth`) and reloaded by the test routine.

## Preprocessing pipeline

Shared across the CNN files (`get_transforms()`):

1. Convert grayscale to 3 channels (models expect RGB input).
2. Resize to `224x224` (`299x299` for Inception V3).
3. `ToTensor`.
4. Normalize with ImageNet mean/std.

The ViT notebook uses `ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")` instead.

## Evaluation

Both train and test loops report:

- Accuracy (subset accuracy on multi-hot labels)
- Precision / Recall (micro)
- F1 (micro and macro)
- Hamming loss
- AUC (where implemented)

The ResNet variants additionally generate Grad-CAM heatmaps written to disk; the ViT notebook produces per-class confusion matrices and ROC curves.

## Helper functions

### `custom_collate_fn(batch)`

DeepLake's default collator stacks tensors per key, but per-image label lists vary in length and can't be stacked. This collator instead returns a dict mapping each key to a Python list of the per-sample values, leaving variable-length labels intact for downstream conversion.

### `convert_labels_to_multihot(raw_labels, num_classes=15)`

Converts each sample's variable-length label list into a fixed-length multi-hot vector — index `i` is `1.0` if class `i` is present in that sample, `0.0` otherwise. This gives the loss function a consistent target tensor shape across the batch.
