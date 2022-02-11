import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.models import vgg16_bn, resnet152
from torch.optim.lr_scheduler import LambdaLR
from p1_dataset import p1_data

# For reproducibility
torch.manual_seed(0)

work_dir = '.'
data_dir = 'hw1_data/p1_data'
root = os.path.join(work_dir, data_dir)

train_tfm = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.RandomResizedCrop(256,scale=(0.8, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
])

test_tfm = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
])

batch_size = 32

# Construct datasets

train_set = p1_data(root, 'train', transform=train_tfm)
valid_set = p1_data(root, 'valid', transform=test_tfm)

# Construct data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# model

model = resnet152(pretrained=True)
# modify the last layer from Linaer(4096, 1000) to (4096, 50)
# model.features = model.features[:24]
# model.classifier = model.classifier[:4]
model.fc = nn.Linear(2048, 50) 

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# put it on the device specified.
model = model.to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# The number of training epochs.
n_epochs = 60
warmup_epoch = 10
lambda0 = lambda epoch: epoch / warmup_epoch if epoch < warmup_epoch else  0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lambda0)

model_dir = os.path.join(work_dir, 'models')
model_path = os.path.join(model_dir, 'model.ckpt')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
best_acc = 0

for epoch in range(n_epochs):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in train_loader:

        # A batch consists of image data and corresponding labels.
        imgs, labels, filename = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

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

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in valid_loader:

        # A batch consists of image data and corresponding labels.
        imgs, labels, filename = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    
    scheduler.step()

    # if improved
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.5f}'.format(best_acc))
print('training is done! get a model with acc {:.5f}'.format(best_acc))
