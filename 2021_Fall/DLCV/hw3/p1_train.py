import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
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

batch_size = 8
workers = 2
lr = 1e-5
weight_decay = 1e-5
num_epochs = 30
num_classes = 37


root = 'hw3_data/p1_data'
model_dir = './p1_models'
os.makedirs(model_dir, exist_ok=True)

train_dir = os.path.join(root, 'train')
valid_dir = os.path.join(root, 'val')

train_tfm = transforms.Compose([
        transforms.RandomRotation(30), 
        transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_tfm = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_dataset = p1_data(train_dir, mode='train', transform=train_tfm)
valid_dataset = p1_data(valid_dir, mode='valid', transform=test_tfm)

# Create the dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes)
model = model.to(device)
print(model)

# Initialize Loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# Training Loop
train_losses = []

for epoch in range(num_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for i, batch in enumerate(train_loader):
        
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        labels = labels.long().to(device)
        
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        
        # Compute the gradients for parameters.
        loss.backward()
        
        # Update the parameters with computed gradients.
        optimizer.step()
        
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
    torch.save(model.state_dict(), os.path.join(model_dir, 'p1_model.pth'))

plt.figure(figsize=(10,5))
plt.title("Training Loss")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./p1_Loss.png")

# Testing
print('Testing started!')
model.eval()
test_hit = 0
for i, batch in enumerate(valid_loader):
    imgs, labels = batch
    with torch.no_grad():
        logits = model(imgs.to(device))

    test_hit += (logits.argmax(dim=-1) == labels.to(device)).sum()

test_acc = test_hit / len(valid_dataset)
print(f"Testing Acc = {test_acc:.4f}")
