import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as pth_transforms
from PIL import Image
from glob import glob
import os
import numpy as np
from tqdm import tqdm

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

# Custom crop transform
def crop_to_multiple(x: int):
    def crop(img: Image.Image):
        width, height = img.size
        new_width = (width // x) * x
        new_height = (height // x) * x
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        return img.crop((left, top, right, bottom))
    return pth_transforms.Lambda(crop)

# Define the transform pipeline
transform = pth_transforms.Compose([
    # crop_to_multiple(14),
    pth_transforms.Resize(size=(1302,1302)),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Custom Dataset
class FungiImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# Load image paths
img_files = glob("FungiImages/fungi_train*.jpg")

# Create dataset and dataloader
dataset = FungiImageDataset(img_files, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

# Example: iterate through the dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_features = []

for img_tensor in tqdm(dataloader):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        features = dinov2_vits14(img_tensor)
    features.cpu()
    dino_features.append(features.cpu().numpy())
np.save("dino_features_crop.npy", np.vstack(dino_features))
