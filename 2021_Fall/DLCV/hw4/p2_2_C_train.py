import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
from p2_dataset import *
from p2_model import *
from office_label_converter import *
import random

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser(description="Self-supervised Learning")
parser.add_argument('--setting', type=str, help='setting (A, B, C, D or E)')

# constants
BATCH_SIZE = 32
IMAGE_SIZE = 128
EPOCHS     = 80
LR         = 3e-4
NUM_WORKERS= 2

args = parser.parse_args()
setting = args.setting
SL = ['B', 'D']
SSL = ['C', 'E']
fixedResnet = setting in ['D', 'E']
model_path = None
if setting in SL:
    model_path = "./hw4_data/pretrain_model_SL.pt"
elif setting in SSL:
    model_path = "./p2_models/p2_model_SSL.pth"
csv_path = "./hw4_data/office/train.csv"
data_dir = "./hw4_data/office/train"
ckpt_dir = "p2_models"

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Create dataset
train_set = OfficeHome(csv_path, data_dir)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

# Create model
resnet = models.resnet50(pretrained=False)
if model_path != None:
    resnet.load_state_dict(torch.load(model_path))
resnet = resnet.to(device)
classifier = Classifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer_res = torch.optim.Adam(resnet.parameters(), lr=LR)
optimizer_clf = torch.optim.Adam(classifier.parameters(), lr=LR)

converter = OfficeLabelConverter()
resnet.train()
classifier.train()

if __name__ == '__main__':
    # Training 
    for epoch in range(EPOCHS):
        train_loss = 0.0
        total_hit, total_num = 0.0, 0.0
        for i, batch in enumerate(train_loader):
            img, label, _ = batch
            logits = classifier(resnet(img.to(device)))
            encoded_label = torch.LongTensor(converter.label_encoder(label)).to(device)
            loss = criterion(logits, encoded_label)
            train_loss += loss.item()
            loss.backward()
            
            optimizer_clf.step()
            optimizer_clf.zero_grad()
            
            if not fixedResnet:
                optimizer_res.step()
                optimizer_res.zero_grad()

            total_hit += torch.sum(torch.argmax(logits, dim=1) == encoded_label).item()
            total_num += img.shape[0]
            
        train_acc = total_hit / total_num
        print('epoch {:>3d} / {}: loss: {:6.4f}, acc {:6.4f}'.format(epoch+1, EPOCHS, train_loss, train_acc))
        torch.save(resnet.state_dict(), os.path.join(ckpt_dir, f'p2_2_{setting}_resnet.pth'))
        torch.save(classifier.state_dict(), os.path.join(ckpt_dir, f'p2_2_{setting}_classifier.pth'))
        
    print("Training completed.")