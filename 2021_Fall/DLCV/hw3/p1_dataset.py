import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset

class p1_data(Dataset):
    def __init__(self, root, mode, transform=None):
        """ Initialize the dataset """
        self.mode = mode
        self.root = root
        self.transform = transform
        # read filenames
        self.filenames = [file for file in os.listdir(root)]         
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filepath = os.path.join(self.root, self.filenames[index])
        img = Image.open(filepath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.mode == 'train' or self.mode == 'valid':
            # extracting label from file name
            label = int(self.filenames[index].split('_')[0])
        
            return img, label
        else:
            return img, self.filenames[index]

    def __len__(self):
        
        return self.len