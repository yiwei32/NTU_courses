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
from p3_model import *

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
svhn_dir = "./hw2_data/digits/svhn"
mnistm_dir = "./hw2_data/digits/mnistm"
usps_dir = "./hw2_data/digits/usps"

source_dir = mnistm_dir
target_dir = usps_dir

workspace_dir = '.'
log_dir = os.path.join(workspace_dir, 'p3_logs')
ckpt_dir = os.path.join(workspace_dir, 'p3_checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
extractor_path = os.path.join(ckpt_dir, 'p3_1_2_extractor.pth')
predictor_path = os.path.join(ckpt_dir, 'p3_1_2_predictor.pth')
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Number of classes
num_classes = 10

# Number of training epochs
num_epochs = 30

train_tfm = transforms.Compose([
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
sourceDataset = digitsDataset(root=source_dir, mode="train", transform=train_tfm)
targetDataset = digitsDataset(root=target_dir, mode="test", transform=test_tfm)


# Create the dataloader
train_loader = torch.utils.data.DataLoader(sourceDataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
test_loader = torch.utils.data.DataLoader(targetDataset, batch_size=batch_size,shuffle=False, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")


# Create models
feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)

# Initialize Loss function
criterion = nn.CrossEntropyLoss()

# Setup Adam optimizers
optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())

# Training Loop

best_acc = 0
train_losses = []

for epoch in range(num_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    feature_extractor.train()
    label_predictor.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for i, batch in enumerate(train_loader):
        
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        labels = labels.long().to(device)
        
        # Forward the data. (Make sure data and model are on the same device.)
        logits = label_predictor(feature_extractor(imgs.to(device)))
        
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        
        # Compute the gradients for parameters.
        loss.backward()
        
        # Update the parameters with computed gradients.
        optimizer_F.step()
        optimizer_C.step()
        
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        
        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    train_losses.append(train_loss)
    # Print the information.
    print(f"[{epoch+1:03d}/{num_epochs:03d}] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    torch.save(feature_extractor.state_dict(), extractor_path)
    torch.save(label_predictor.state_dict(), predictor_path)

plt.figure(figsize=(10,5))
plt.title("M->U Training Loss")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./p3_logs/3_1_2_Loss.png")

# Testing
print('Testing started!')
feature_extractor.eval()
label_predictor.eval()
test_accs = []
for i, batch in enumerate(test_loader):
    imgs, labels = batch
    with torch.no_grad():
        logits = label_predictor(feature_extractor(imgs.to(device)))

    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    test_accs.append(acc)

test_acc = sum(test_accs) / len(test_accs)
print(f"Source: MNIST-M, Target: USPS, Acc = {test_acc:.5f}")

