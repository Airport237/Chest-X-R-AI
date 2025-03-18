import load_data
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ChestXRay14(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

        train_loader, ds_train, ds_test = load_data.get_data()
        images = ds_train.images.numpy()
        labels = ds_train.findings.numpy()

        for i, image in enumerate(images):
            self.samples.append({'images': image, 'labels': labels[i]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['images'])
        lable = sample['labels']

        if self.transform:
            image = self.transform(image)

        return {'image': image,
                'label': torch.tensor(lable, dtype=torch.float32),}





