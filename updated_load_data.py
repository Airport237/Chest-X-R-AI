import deeplake
from torchvision import transforms

def get_data():
    print("Loading dataset...")
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')

    print("Creating train loader...")
    train_loader = ds_train.pytorch(
        num_workers=2,
        batch_size=32,
        shuffle=True,
        decode_method={"images": "pil"}
    )

    print("Dataset loaded successfully!")
    return train_loader, ds_train, ds_test

def transform():
    trans = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trans