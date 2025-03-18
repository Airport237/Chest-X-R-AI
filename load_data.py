import deeplake
from torchvision import transforms


def get_data():
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
    train_loader = ds_train.pytorch(num_workers=0, batch_size=4, shuffle=False)
    return train_loader, ds_train, ds_test



def transform():
    trans = transforms.Compose([
    #Need to evaluate this pre processing, this is a first draft to try and get a model working
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trans

