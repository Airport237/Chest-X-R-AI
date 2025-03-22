import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import Loading_data
from tqdm import tqdm
import numpy as np

class ChestXRay14(Dataset):
    def __init__(self, transform=None, limit=35000):
        print("Initializing ChestXRay14 dataset...")
        self.transform = transform
        self.samples = []

        print("Loading dataset from Deep Lake...")
        train_loader, ds_train, ds_test = Loading_data.get_data()
        print("Dataset loaded successfully into Preprocessing_data.py!")

        print(f"\U0001F4F8 Extracting image paths (limited to {limit})...")

        for i in tqdm(range(min(len(ds_train), limit)), desc="Processing images", unit="img"):
            image = ds_train.images[i].numpy(aslist=True)
            label = ds_train.findings[i].numpy(aslist=True)
            self.samples.append({'images': image, 'labels': label})

        print(f"Total Samples Loaded: {len(self.samples)}")
        print("Dataset successfully processed for PyTorch!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_data = sample['images']
        label = sample['labels']

        if image_data.ndim < 2:
            print(f"❌ Skipping corrupted image at index {idx}")
            return self.__getitem__((idx + 1) % len(self.samples))

        if image_data.ndim == 2:
            image_data = np.stack([image_data] * 3, axis=-1)
        elif image_data.ndim == 3:
            if image_data.shape[2] == 1:
                image_data = np.repeat(image_data, 3, axis=2)
            elif image_data.shape[2] == 4:
                image_data = image_data[:, :, :3]

        try:
            image = Image.fromarray(image_data.astype('uint8'))
        except Exception as e:
            print(f"❌ Failed to convert image at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            image = self.transform(image)

        label = np.array(label)
        if label.ndim == 0:
            label = np.zeros(14)
        elif label.size != 14:
            print(f"⚠️ Invalid label size {label.size} at index {idx}, fixing...")
            fixed = np.zeros(14)
            fixed[:min(len(label), 14)] = label[:min(len(label), 14)]
            label = fixed

        label = torch.tensor(label, dtype=torch.float32)

        return {'image': image, 'label': label}

def transform():
    print("Applying transformations to dataset...")
    trans = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    print("Transformations applied successfully!")
    return trans

if __name__ == "__main__":
    print("Running Preprocessing_data.py...")
    dataset = ChestXRay14(transform=transform(), limit=35000)
    print(f"Dataset successfully instantiated with {len(dataset)} samples.")
