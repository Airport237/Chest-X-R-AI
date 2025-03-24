import deeplake

def get_data():
    """
    Loads the training and test datasets from DeepLake and creates a PyTorch DataLoader
    for the training data.
    """
    print("Starting the dataset loading process...")

    try:
        # Load the training dataset
        print("Loading the training dataset from DeepLake...")
        ds_train = deeplake.load('hub://activeloop/nih-chest-xray-train')
        print("Training dataset loaded successfully!")

        # Load the test dataset
        print("Loading the test dataset from DeepLake...")
        ds_test = deeplake.load('hub://activeloop/nih-chest-xray-test')
        print("Test dataset loaded successfully!")

        # Print out the number of samples in each dataset
        print(f"Total training samples: {len(ds_train)}")
        print(f"Total test samples: {len(ds_test)}")

        # Create a PyTorch DataLoader for the training dataset
        print("Creating the training DataLoader...")
        train_loader = ds_train.pytorch(
            num_workers=2,
            batch_size=32,
            shuffle=True,
            decode_method={"images": "pil"}
        )
        print("Training DataLoader created successfully!")

        return train_loader, ds_train, ds_test

    except Exception as e:
        print(f"An error occurred while loading the datasets: {e}")

if __name__ == "__main__":
    # Execute the data loading function and observe progress in the console
    get_data()