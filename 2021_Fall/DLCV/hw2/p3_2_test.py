import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import sys
from p3_model import *
from PIL import Image

# Set random seed for reproducibility
manualSeed = 0
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 32
workers = 2
test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class digitsDataset(Dataset):
    def __init__(self, root, transform):
        """ Initialize the dataset """
        self.root = root
        self.filenames = None
        self.transform = transform
        
        # read images names
        self.filenames = sorted([file for file in os.listdir(root) if file.endswith('.png')])
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        filepath = os.path.join(self.root, self.filenames[index])
        image = Image.open(filepath).convert('RGB')
        image = self.transform(image)
        
        return image, self.filenames[index]
    
    def __len__(self):
        return self.len

target_dir = sys.argv[1]
target = sys.argv[2]
output_path = sys.argv[3]

source = None

if target == "mnistm":
    source = "svhn"
elif target == "usps":
    source = "mnistm"
elif target == "svhn":
    source = "usps"

extractor_path = "./p3_2_" + target + "_extractor_model.pth"
predictor_path = "./p3_2_" + target + "_predictor_model.pth"

targetDataset = digitsDataset(root=target_dir, transform=test_tfm)
test_dataloader = torch.utils.data.DataLoader(targetDataset, batch_size=batch_size,shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu" 

label_predictor = LabelPredictor().to(device)
feature_extractor = FeatureExtractor().to(device)
label_predictor.load_state_dict(torch.load(predictor_path))
feature_extractor.load_state_dict(torch.load(extractor_path))

label_predictor.eval()
feature_extractor.eval()


predictions = []
filenames = []

for i, (test_data, filename) in enumerate(test_dataloader):
    test_data = test_data.to(device)  
    with torch.no_grad():
        class_logits = label_predictor(feature_extractor(test_data))
    predictions.extend(class_logits.argmax(dim=-1).cpu().numpy().tolist())
    filenames.extend(filename)


with open(output_path, "w") as f:
    
    # The first row must be "Id, Category"
    f.write("image_name,label\n")
    
    # For the rest of the rows, each image id corresponds to a predicted class.
    for fn, pred in  zip(filenames, predictions):
        f.write(f"{fn},{pred}\n")
