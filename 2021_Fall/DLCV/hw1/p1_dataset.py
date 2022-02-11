import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset

class p1_data(Dataset):
    def __init__(self, root, mode, transform=None):
        """ Initialize the dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.mode = mode
        self.root = root
        self.transform = transform
        
        # read filenames
        if self.mode == 'train':
            for img_class in range(50):
                for idx in range(450):
                    filename = os.path.join(root, 'train_50', str(img_class)+'_'+str(idx)+'.png')
                    self.filenames.append((filename, img_class))
        
        elif self.mode == 'valid':
            for img_class in range(50):
                for idx in range(450, 500):
                    filename = os.path.join(root, 'val_50', str(img_class)+'_'+str(idx)+'.png')
                    self.filenames.append((filename, img_class))
        else:
            # testing
            test_path_list = sorted([file for file in os.listdir(root) if file.endswith('.png')])
            for test_path in test_path_list:
                filename = os.path.join(root, test_path)
                self.filenames.append((filename, 0)) # 0 for fake labels            
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_full_fn, label = self.filenames[index]
        image = Image.open(image_full_fn)
        
        if self.transform is not None:
            image = self.transform(image)

        image_fn = os.path.relpath(image_full_fn, self.root)

        return image, label, image_fn
    
    def __len__(self):
        
        return self.len
