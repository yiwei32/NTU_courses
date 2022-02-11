import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from mean_iou_evaluate import read_masks

class p2_data(Dataset):
    def __init__(self, root, mode, transform=None):
        """ Initialize the dataset """
        self.images = []
        self.labels = []
        self.filenames = []
        self.mode = mode
        self.root = root
        self.transform = transform

        # read filenames
        if self.mode == 'train':
            for idx in range(2000):
                number = str(idx).zfill(4)
                sat_filename = os.path.join(root, 'train', number +'_sat.jpg')
                mask_filename = os.path.join(root, 'train', number +'_mask.png')
                self.filenames.append((sat_filename, mask_filename))
                sat_image = Image.open(sat_filename)

                if self.transform is not None:
                    sat_image = self.transform(sat_image)

                self.images.append(sat_image)
            self.labels = read_masks(os.path.join(root, 'train'))

        elif self.mode == 'valid':
            for idx in range(257):
                number = str(idx).zfill(4)
                sat_filename = os.path.join(root, 'validation', number +'_sat.jpg')
                mask_filename = os.path.join(root, 'validation', number +'_mask.png')
                self.filenames.append((sat_filename, mask_filename))
                sat_image = Image.open(sat_filename)

                if self.transform is not None:
                    sat_image = self.transform(sat_image)

                self.images.append(sat_image)
            self.labels = read_masks(os.path.join(root, 'validation')) 
        else:
            # testing
            test_path_list = sorted([file for file in os.listdir(root) if file.endswith('.jpg')])
            for test_path in test_path_list:
                sat_filename = os.path.join(root, test_path)
                self.filenames.append((sat_filename, 0)) # 0 for fake mask
                sat_image = Image.open(sat_filename)
                
                if self.transform is not None:                                                                         sat_image = self.transform(sat_image)
               
                self.images.append(sat_image)
                self.labels.append(0)
                
     
        self.len = len(self.filenames)
        
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        
        image_full_fn = self.filenames[index][0]
        sat_image = self.images[index]
        label = self.labels[index]

        
        fileidx = os.path.relpath(image_full_fn, self.root)[:4]
         
        return sat_image, label, fileidx
    
    def __len__(self):
        
        return self.len
