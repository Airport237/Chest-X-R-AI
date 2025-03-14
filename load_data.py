import deeplake
import torch

def get_data():
    ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
    ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
    dataloader = ds_train.pytorch(num_workers=0, batch_size=4, shuffle=False)
    return dataloader, ds_train, ds_test