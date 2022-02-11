import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.filenames = np.array(self.data_df["filename"])

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = Image.open(os.path.join(self.data_dir, filename))
        return self.transform(image)
    
    def __len__(self):
        return len(self.filenames)
    
class OfficeHome(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.filenames = np.array(self.data_df["filename"])
        self.labels = np.array(self.data_df["label"])

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        image = Image.open(os.path.join(self.data_dir, filename))
        return self.transform(image), label, filename
    
    def __len__(self):
        return len(self.filenames)
    
