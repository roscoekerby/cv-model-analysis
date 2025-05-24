# utils/dataset_loader.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        """
        Custom dataset for image classification from folder structure:
        root_dir/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image3.jpg
            └── image4.jpg
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []

        # Get all class directories
        if os.path.isdir(root_dir):
            self.classes = sorted([d for d in os.listdir(root_dir)
                                   if os.path.isdir(os.path.join(root_dir, d))])

        if class_to_idx is None:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        # Collect all image paths and labels
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for filename in os.listdir(class_dir):
                if filename.lower().endswith(supported_extensions):
                    self.samples.append((os.path.join(class_dir, filename), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset(name, custom_path=None, image_size=(224, 224)):
    """
    Load dataset - either built-in or custom
    Args:
        name: Dataset name ('CIFAR-10', 'CIFAR-100', 'Custom')
        custom_path: Path to custom dataset folder (required if name='Custom')
        image_size: Tuple of (width, height) for image resizing
    """
    if name == "CIFAR-10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        loader = DataLoader(test_set, batch_size=1, shuffle=False)
        images, labels = [], []
        for img, lbl in loader:
            images.append(img[0])
            labels.append(lbl[0])
        return images, labels

    elif name == "CIFAR-100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        loader = DataLoader(test_set, batch_size=1, shuffle=False)
        images, labels = [], []
        for img, lbl in loader:
            images.append(img[0])
            labels.append(lbl[0])
        return images, labels

    elif name == "Custom":
        if not custom_path or not os.path.exists(custom_path):
            raise ValueError("Custom dataset path must be provided and must exist.")

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        custom_dataset = CustomImageDataset(custom_path, transform=transform)
        loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

        images, labels = [], []
        for img, lbl in loader:
            images.append(img[0])
            labels.append(lbl[0])

        print(f"Loaded custom dataset with {len(images)} images from {len(custom_dataset.classes)} classes:")
        print(f"Classes: {custom_dataset.classes}")

        return images, labels

    else:
        raise ValueError(f"Dataset '{name}' not supported. Available: CIFAR-10, CIFAR-100, Custom")

