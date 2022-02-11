import torch
import torch.nn as nn
from torchvision.models import resnet152
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd  
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from p1_dataset import p1_data
import matplotlib.pyplot as plt

model_path = './p1_model.ckpt'

root = './hw1_data/p1_data/'

tfm = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
])
batch_size = 32
    
# construct dataset and dataloader

valid_set = p1_data(root, 'valid', transform=tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = resnet152(pretrained=False)
model.fc = nn.Linear(2048, 50)
model.load_state_dict(torch.load(model_path))
model.fc = nn.Identity()
model = model.to(device)

X = torch.FloatTensor()
X = X.to(device)
Y = torch.LongTensor()
Y = Y.to(device)
for batch in valid_loader:
    imgs, labels, fns = batch
    with torch.no_grad():
        outputs = model(imgs.to(device))
    X = torch.cat((X, outputs.to(device)), 0).to(device)
    Y = torch.cat((Y, labels.to(device)), 0).to(device) 

X = X.cpu().numpy()
Y = Y.cpu().numpy()

tsne = TSNE(n_components=2, verbose=1, random_state=0)
X_tsne = tsne.fit_transform(X)

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))

for i in range(X_norm.shape[0]):
    fig = plt.scatter(X_norm[i, 0], X_norm[i, 1], marker='o', color=plt.cm.Set1(Y[i]/50))
fig = fig.get_figure()
fig.savefig("T-SNE.png")
