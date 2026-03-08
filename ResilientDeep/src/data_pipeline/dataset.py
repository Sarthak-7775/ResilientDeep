# hereeee --Basic preprocessing of images --- fixing sizes, 
# imge, normalization, etc. for ResNet architectures.

import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class CelebDFDataset(Dataset):
    def __init__(self, root_dir, real_folder="Celeb-real", fake_folder="Celeb-synthesis", transform=None):
        """
        Initializes the dataset and pre-computes paths for O(1) access.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Map folders to binary labels (0 for Real, 1 for Fake)
        classes = {real_folder: 0, fake_folder: 1}
        
        for folder_name, label in classes.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found -> {folder_path}")
                continue
                
            for img_name in os.listdir(folder_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # O(1) retrieval
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Standard preprocessing for ResNet architectures
baseline_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])