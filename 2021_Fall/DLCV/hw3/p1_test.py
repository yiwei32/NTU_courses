import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import sys
import random
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_vit import ViT
from p1_dataset import p1_data

# Set random seed for reproducibility
manualSeed = 0
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

test_path = sys.argv[1]
output_path = sys.argv[2]
model_path = './p1_model.pth'

batch_size = 8
workers = 2
num_classes = 37

test_tfm = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_dataset = p1_data(test_path, mode='test', transform=test_tfm)

# Create the dataloader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

predictions = []
filenames = []

for i, (test_data, filename) in enumerate(test_loader):
    test_data = test_data.to(device)  
    with torch.no_grad():
        class_logits = model(test_data)
    predictions.extend(class_logits.argmax(dim=-1).cpu().numpy().tolist())
    filenames.extend(filename)

with open(output_path, "w") as f:
    f.write("filename,label\n")
    # For the rest of the rows, each image name corresponds to a predicted class.
    for fn, pred in  zip(filenames, predictions):
        f.write(f"{fn},{pred}\n")
