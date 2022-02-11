import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader 
from p2_dataset import p2_data
from p2_model import FCN32s, FCN8s
import numpy as np
import sys
import os
from mean_iou_evaluate import mean_iou_score, read_masks 
import viz_mask
from PIL import Image


model_path = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

batch_size = 1

# transforms
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# construct dataset

test_set = p2_data(input_dir, 'test', transform=tfm)

# construct data loader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = FCN8s()
model.load_state_dict(torch.load(model_path))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

pred = torch.FloatTensor()
pred = pred.to(device)
filenames = []
for batch in test_loader:
    imgs, labels, fns = batch
    with torch.no_grad():
        outputs = model(imgs.to(device))
    pred = torch.cat((pred, outputs.to(device)), 0).to(device)
    filenames.append(fns)
    
pred = pred.cpu().numpy()
pred = np.argmax(pred,1)


n_masks = test_set.len
masks_RGB = np.empty((n_masks, 512, 512, 3))
for i, p in enumerate(pred):
    masks_RGB[i, p == 0] = [0,255,255] # (Cyan: 011) Urban land
    masks_RGB[i, p == 1] = [255,255,0] # (Yellow: 110) Agriculture land
    masks_RGB[i, p == 2] = [255,0,255] # (Purple: 101) Rangeland
    masks_RGB[i, p == 3] = [0,255,0] # (Green: 010) Forest land
    masks_RGB[i, p == 4] = [0,0,255] # (Blue: 001) Water
    masks_RGB[i, p == 5] = [255,255,255] # (White: 111) Barren land
    masks_RGB[i, p == 6] = [0,0,0] # (Black: 000) Unknown
    
masks_RGB = masks_RGB.astype(np.uint8)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("path:",os.path.join(output_dir, "XXXX_mask.png"))
for i, mask_RGB in enumerate(masks_RGB):
    img = Image.fromarray(mask_RGB)
    img.save(os.path.join(output_dir, filenames[i][0] + "_mask.png"))

