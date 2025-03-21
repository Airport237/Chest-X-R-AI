import deeplake


def get_data():
    print("Starting dataset loading...")

    try:
        print("Attempting to load training dataset...")
        ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
        print("Training dataset loaded!")

        print("Attempting to load test dataset...")
        ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
        print("Test dataset loaded!")

        print(f"Number of training samples: {len(ds_train)}")
        print(f"Number of test samples: {len(ds_test)}")

        print("Creating train DataLoader...")
        train_loader = ds_train.pytorch(
            num_workers=2,
            batch_size=32,
            shuffle=True,
            decode_method={"images": "pil"}
        )
        print("Train DataLoader created!")

        return train_loader, ds_train, ds_test

    except Exception as e:
        print(f"Error occurred while loading dataset: {e}")


if __name__ == "__main__":
    get_data()