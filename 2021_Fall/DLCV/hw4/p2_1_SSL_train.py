import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from byol_pytorch import BYOL
from torchvision import models
from p2_dataset import *
import random

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

# constants
BATCH_SIZE = 32
IMAGE_SIZE = 128
EPOCHS     = 100
LR         = 3e-4
NUM_WORKERS= 2

root_dir = "./hw4_data/mini"
model_dir = "p2_models"
os.makedirs(model_dir, exist_ok=True)
data_dir = os.path.join(root_dir, 'train')
csv_path = os.path.join(root_dir, 'train.csv')

if __name__ == '__main__':
    
    train_set = MiniDataset(csv_path, data_dir)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    
    resnet = models.resnet50(pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    learner = BYOL(
    resnet,
    image_size = IMAGE_SIZE,
    hidden_layer = 'avgpool'
    )
    learner.to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=LR)
    
    learner.train()
    for epoch in range(EPOCHS):
        
        for i, data in enumerate(train_loader):
            data = data.to(device)
            loss = learner(data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder

        print('epoch {} / {}, loss={:.4f}'
                    .format(epoch+1, EPOCHS, loss.item()))
        torch.save(resnet.state_dict(), os.path.join(model_dir, 'p2_model_SSL.pth'))
        # lr_scheduler.step()
    
    print("Training completed.")