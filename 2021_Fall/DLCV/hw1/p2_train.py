import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from p2_dataset import p2_data
from p2_model import FCN32s, FCN8s
from mean_iou_evaluate import mean_iou_score, read_masks

# For reproducibility
torch.manual_seed(0)

work_dir = '.'
data_dir = 'hw1_data/p2_data'
root = os.path.join(work_dir, data_dir)

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
])

batch_size = 16

# Construct datasets

train_set = p2_data(root, 'train', transform=tfm)
valid_set = p2_data(root, 'valid', transform=tfm)

# Construct data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# model

model = FCN8s()
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# put it on the device specified.
model = model.to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.NLLLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# The number of training epochs.
n_epochs = 25
warmup_epoch = 5
lambda0 = lambda epoch: epoch / warmup_epoch if epoch < warmup_epoch else  0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lambda0)

model_dir = os.path.join(work_dir, 'p2_models')
model_path = os.path.join(model_dir, 'p2_FCN8s_model.ckpt')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    

best_iou = 0

for epoch in range(n_epochs):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []

    # Iterate the training set by batches.
    for batch in train_loader:

        # A batch consists of image data and corresponding labels.
        imgs, labels, filename = batch
        labels = torch.Tensor.long(labels)
        # Forward the data. (Make sure data and model are on the same device.)
        outputs = model(imgs.to(device)) 
        outputs = F.log_softmax(outputs, dim= 1)
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(outputs, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        # acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        # train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    # train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    pred = torch.FloatTensor()
    pred = pred.to(device)
    gt = torch.Tensor()
    gt = gt.to(device) 

    # Iterate the validation set by batches.
    for batch in valid_loader:

        # A batch consists of image data and corresponding labels.
        imgs, labels, filename = batch
        labels = torch.Tensor.long(labels)        
        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          outputs = model(imgs.to(device))
        outputs = F.log_softmax(outputs, dim= 1)
        pred = torch.cat((pred,outputs.to(device)),0).to(device)
        gt = torch.cat((gt, labels.to(device)), 0).to(device)
        # We can still compute the loss (but not the gradient).
        
        loss = criterion(outputs, labels.to(device))

        # Compute the accuracy for current batch.
        # acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        # valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}")
    
    scheduler.step()

    pred = pred.cpu().numpy()
    pred = np.argmax(pred,1)
    gt = gt.cpu().numpy()
    mean_iou = mean_iou_score(pred, gt)
    
    # record model progress
    if epoch in [1, 11, 21]:
        progress_dir = "./progress"
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir)
        # save 0010_mask, 0097_mask, and 0107_mask
        ids = [10, 97, 107]
        imgs = torch.tensor([valid_set.images[i].cpu().detach().numpy() for i in ids])
        
        with torch.no_grad():
            outputs = model(imgs.to(device))
            
        outputs = outputs.cpu().numpy()
        outputs = np.argmax(outputs, 1)
        
        n_masks = len(imgs)
        masks_RGB = np.empty((n_masks, 512, 512, 3))
        for i, p in enumerate(outputs):
            masks_RGB[i, p == 0] = [0,255,255] # (Cyan: 011) Urban land
            masks_RGB[i, p == 1] = [255,255,0] # (Yellow: 110) Agriculture land
            masks_RGB[i, p == 2] = [255,0,255] # (Purple: 101) Rangeland
            masks_RGB[i, p == 3] = [0,255,0] # (Green: 010) Forest land
            masks_RGB[i, p == 4] = [0,0,255] # (Blue: 001) Water
            masks_RGB[i, p == 5] = [255,255,255] # (White: 111) Barren land
            masks_RGB[i, p == 6] = [0,0,0] # (Black: 000) Unknown
    
        masks_RGB = masks_RGB.astype(np.uint8)

        for id, mask_RGB in zip(ids, masks_RGB):
            img = Image.fromarray(mask_RGB)
            img.save(os.path.join(progress_dir, str(epoch) + '_' + str(id).zfill(4) + '_mask.png'))

    
    # if improved
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), model_path)
        print('saving model with mean_iou {:.5f}'.format(best_iou))
print('training is done! get a model with mean_iou {:.5f}'.format(best_iou))
