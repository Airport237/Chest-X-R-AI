import torch
import torch.nn as nn
import torch.optim as optim
import deeplake
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models

from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score
from AlexNET_CNN_Model import custom_collate_fn, custom_get_data, get_transforms, train_model, test


def main():
    print("========== Starting Model Training Process ==========")
    train_loader, test_loader, ds_train, ds_test = custom_get_data()
    transform_pipeline = get_transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_classes = 15
    model = models.vgg16(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    model.to(device)
    train_model(model, train_loader, transform_pipeline, device, num_epochs=5)
    print("========== Model Training is Complete ==========")
    test(model, test_loader, transform_pipeline, device, saved_model="model.pth")



if __name__ == "__main__":
    main()
'''
=============================================
Recall (micro): 0.4829
F1 Score (micro): 0.5285
F1 Score (macro): 0.0491
Hamming Loss: 0.0694
Accuracy: 0.5831
=============================================
=============================================
Testing Finished
Precision (micro): 0.3550
Recall (micro): 0.2459
F1 Score (micro): 0.2905
F1 Score (macro): 0.0349
Hamming Loss: 0.1156
Accuracy: 0.3550
=============================================
'''