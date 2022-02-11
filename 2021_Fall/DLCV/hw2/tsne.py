# tsne
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from PIL import Image
from sklearn.manifold import TSNE
import sys
from p3_model import FeatureExtractor

target = sys.argv[1]
source = None
model_path = None
if target == "mnistm":
    source = "svhn"
elif target == "usps":
    source = "mnistm"
elif target == "svhn":
    source = "usps"

# Number of smaples
num_samples = 500
    
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Number of classes
num_classes = 10

# Number of hidden dimension
hidden_dims = 128

# Root directory for dataset
root = "./hw2_data/digits"
source_dir = os.path.join(root, source)
target_dir = os.path.join(root, target)

workspace_dir = '.'
log_dir = os.path.join(workspace_dir, 'p3_logs')
model_path = "./p3_2_" + target + "_extractor_model.pth"

test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class digitsDataset(Dataset):
    def __init__(self, root, mode, transform):
        """ Initialize the dataset """
        self.root = root
        self.labels = []
        self.filenames = []
        self.mode = mode
        self.transform = transform 
        
        # read images names
        data_dir = os.path.join(root, mode)
        self.filenames = sorted([file for file in os.listdir(data_dir) if file.endswith('.png')])
        
        #read labels
        labelpath = os.path.join(root, mode+'.csv')
        labels = pd.read_csv(labelpath).iloc[:, 1]
        self.labels = torch.Tensor(labels)
        
        self.len = len(labels)
    def __getitem__(self, index):
        data_dir = os.path.join(self.root, self.mode)
        filepath = os.path.join(data_dir, self.filenames[index])
        image = Image.open(filepath).convert('RGB')
        image = self.transform(image)
        
        return image, self.labels[index]
    
    def __len__(self):
        return self.len

# Create dataset
sourceDataset = digitsDataset(root=source_dir, mode="test", transform=test_tfm)
targetDataset = digitsDataset(root=target_dir, mode="test", transform=test_tfm)

# Create the dataloader
source_dataloader = torch.utils.data.DataLoader(sourceDataset, batch_size=batch_size, shuffle=True, num_workers=workers)
target_dataloader = torch.utils.data.DataLoader(targetDataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FeatureExtractor().to(device)
model.load_state_dict(torch.load(model_path))

X = torch.FloatTensor()
X = X.to(device)
Y_class = torch.LongTensor()
Y_domain = torch.LongTensor()
Y_class = Y_class.to(device)
Y_domain = Y_domain.to(device)
for i in range(num_samples):
    source_img, source_label = sourceDataset[i]
    target_img, target_label = targetDataset[i]
    source_img = torch.unsqueeze(source_img, 0)
    target_img = torch.unsqueeze(target_img, 0)
    source_label = torch.unsqueeze(source_label, 0)
    target_label = torch.unsqueeze(target_label, 0)  
    mixed_data = torch.cat((source_img, target_img), dim=0)
    mixed_labels = torch.cat((source_label, target_label), dim=0)
    domain_labels = torch.Tensor([[0],[1]])
                              
    with torch.no_grad():
        feature = model(mixed_data.to(device))
    X = torch.cat((X, feature.to(device)), dim=0)
    Y_class = torch.cat((Y_class, mixed_labels.to(device)), dim=0) 
    Y_domain = torch.cat((Y_domain, domain_labels.to(device)), dim=0)

X = X.cpu().numpy()
Y_class = Y_class.cpu().numpy()
Y_domain = Y_domain.cpu().numpy()

tsne = TSNE(n_components=2, verbose=1, random_state=0)
X_tsne = tsne.fit_transform(X)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

fig = plt.figure(figsize=(20, 8))
ax_class = fig.add_subplot(1,2,1)
ax_class.set_title("Digit Class")  
ax_domain = fig.add_subplot(1,2,2)
ax_domain.set_title(f"{source} / {target} Domain") 
for i in range(X_norm.shape[0]):
    ax_class.scatter(X_norm[i, 0], X_norm[i, 1], marker='o', color=plt.cm.Set1(Y_class[i]/10))
    ax_domain.scatter(X_norm[i, 0], X_norm[i, 1], marker='o', color=plt.cm.Set1(Y_domain[i]))

# fig_class = fig_class.get_figure()
fig.savefig(os.path.join(log_dir, f"target={target}_T-SNE.png"))                

