import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch
import numpy as np
import random

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.labels = np.array(self.data_df["label"])
        self.filenames = np.array(self.data_df["filename"])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        image = Image.open(os.path.join(self.data_dir, filename))
        image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.filenames)
    
class CategoriesSampler(Sampler):
    
    def __init__(self, labels, classes_per_it, num_per_class, episodes):
        '''
        Args:
            - labels: an iterable containing all the labels for the current dataset
            samples indexes will be infered from this iterable.
            - classes_per_it: number of random classes for each iteration
            - num_per_class: number of samples for each iteration for each class (support + query)
            - episodes: number of iterations (episodes) per epoch
        '''
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.num_per_class = num_per_class
        self.episodes = episodes
        
        class_names = np.unique(labels)
        self.idxs_per_class = [] # e.g. [[0,1,2],[3,4],[5,6,7], ... ]
        for c in class_names:
            index_slice = np.argwhere(labels == c).reshape(-1)
            index_slice = torch.from_numpy(index_slice)
            self.idxs_per_class.append(index_slice)
        

    def __iter__(self):
        
        for i_batch in range(self.episodes):
            batch = []
            sample_classes = torch.randperm(len(self.idxs_per_class))[:self.classes_per_it] # e.g. [3,6,7,10,1]
            for c in sample_classes:
                sample_class = self.idxs_per_class[c]
                # random permute then select first num_per_class samples
                pos = torch.randperm(len(sample_class))[:self.num_per_class] 
                batch.append(sample_class[pos])
            # print(batch)
            batch = torch.stack(batch).t().reshape(-1)
            # print(batch)
            yield batch
            

    def __len__(self):
        return self.episodes