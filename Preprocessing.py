import torchvision.transforms as transforms


def get_transforms():
    """
    Build a preprocessing pipeline that:
      1. Resizes images to 224x224 pixels.
      2. Converts images to PyTorch tensors.
      3. Normalizes images with ImageNet mean and standard deviation.
    """

    # Step 1: Resize the image
    print("Resizing images to 224x224 pixels...")
    resize_transform = transforms.Resize((224, 224))

    # Step 2: Convert image to Tensor
    print("Converting images to PyTorch tensors...")
    tensor_transform = transforms.ToTensor()

    # Step 3: Normalize the image
    print("Normalizing images with ImageNet mean and std values...")
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    # Combine all transforms into a single pipeline
    print("Combining all transformations into one pipeline...")
    transform_pipeline = transforms.Compose([
        resize_transform,
        tensor_transform,
        normalize_transform
    ])

    print("Preprocessing pipeline is ready to use.")
    return transform_pipeline


if __name__ == "__main__":
    print("Testing the preprocessing pipeline...")
    pipeline = get_transforms()
    print("Here is your pipeline:")
    print(pipeline)