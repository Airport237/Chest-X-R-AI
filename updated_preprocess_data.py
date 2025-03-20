import torch
from torch.utils.data import Dataset
from PIL import Image
import load_data

class ChestXRay14(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

        print("Loading dataset inside ChestXRay14...")
        train_loader, ds_train, ds_test = load_data.get_data()

        print("Extracting image paths...")
        images = ds_train.images.numpy(aslist=True)
        labels = ds_train.findings.numpy(aslist=True)

        for i, image in enumerate(images):
            self.samples.append({'images': image, 'labels': labels[i]})

        print("Dataset successfully loaded into ChestXRay14!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['images']
        label = sample['labels']

        print(f"Loading image {idx}: {image_path}")  # Debugging step

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)

        return {'image': image, 'label': label}