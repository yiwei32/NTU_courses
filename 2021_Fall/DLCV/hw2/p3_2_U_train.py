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
from p3_model import *

# Set random seed for reproducibility
manualSeed = 0
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

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Number of classes
num_classes = 10

# Number of training epochs
num_epochs = 30

# Coefficient for domain loss
lamb = 0.1

# Hidden dimension of the output of feature extractor
hidden_dims = 128

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

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
targetDataset = digitsDataset(root=target_dir, mode="train", transform=train_tfm)

# Create the dataloader
source_dataloader = torch.utils.data.DataLoader(sourceDataset, batch_size=batch_size, shuffle=True, num_workers=workers)
target_dataloader = torch.utils.data.DataLoader(targetDataset, batch_size=batch_size,shuffle=True, num_workers=workers)
    

feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)  
domain_classifier = DomainClassifier().to(device)  

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())


def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 調控adversarial的loss係數。
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.to(device)  
        source_label = source_label.long().to(device)  
        target_data = target_data.to(device)  
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)  
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Label Predictor
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num


for epoch in range(num_epochs):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=lamb)

    torch.save(feature_extractor.state_dict(), os.path.join(ckpt_dir, f'p3_2_usps_extractor_model.pth'))
    torch.save(label_predictor.state_dict(), os.path.join(ckpt_dir, f'p3_2_usps_predictor_model.pth'))

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch+1, train_D_loss, train_F_loss, train_acc))


test_dataset = digitsDataset(root=target_dir, mode="test", transform=test_tfm)
test_dataloader = torch.utils.data.DataLoader(targetDataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
test_accs = []
label_predictor.eval()
feature_extractor.eval()

for i, (test_data, labels) in enumerate(test_dataloader):
    test_data = test_data.to(device)  

    class_logits = label_predictor(feature_extractor(test_data))

    acc = (class_logits.argmax(dim=-1) == labels.to(device)).float().mean()
    test_accs.append(acc)

test_acc = sum(test_accs) / len(test_accs)
print(f"Source: MNIST-M, Target: USPS, Acc = {test_acc:.5f}")

