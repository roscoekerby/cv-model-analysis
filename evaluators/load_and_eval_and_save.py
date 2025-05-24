import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# === CONFIG ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
METADATA_PATH = r"C:\Users\roscoe\PycharmProjects\skinsight-models\datasets\PAD-UFES-20\metadata.csv"
IMAGE_DIR = r"C:\Users\roscoe\PycharmProjects\skinsight-models\datasets\PAD-UFES-20\All Images"
STATE_DICT_PATH = r"C:\roscoekerby GitHub\cv-model-analysis\models\final_model (1).pth"
FULL_MODEL_SAVE_PATH = r"C:\roscoekerby GitHub\cv-model-analysis\models\vit_dermatology_full.pth"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load metadata and labels ===
metadata_df = pd.read_csv(METADATA_PATH)
class_names = sorted(metadata_df['diagnostic'].dropna().unique())
label_to_index = {name: idx for idx, name in enumerate(class_names)}

filename_to_label = {
    row['img_id']: label_to_index[row['diagnostic']]
    for _, row in metadata_df.iterrows()
    if row['diagnostic'] in label_to_index
}

filenames = list(filename_to_label.keys())
labels = list(filename_to_label.values())

train_fns, val_fns, train_labels, val_labels = train_test_split(
    filenames, labels, test_size=0.2, stratify=labels, random_state=42
)

# === Custom Dataset ===
class DermatologyDataset(Dataset):
    def __init__(self, filenames, labels, image_dir, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# === Transform and Loader ===
val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_dataset = DermatologyDataset(val_fns, val_labels, IMAGE_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model Setup ===
NUM_CLASSES = len(class_names)

# Suppress deprecation warnings by using weights=None instead of pretrained=False
model = vit_b_16(weights=None)
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Inference ===
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

balanced_acc = balanced_accuracy_score(all_labels, all_preds)
print(f"✅ Balanced Accuracy (GPU/CPU): {balanced_acc:.4f}")

# === Optional: Save Full Model for Future Use ===
torch.save(model, FULL_MODEL_SAVE_PATH)
print(f"📦 Full model saved to: {FULL_MODEL_SAVE_PATH}")
