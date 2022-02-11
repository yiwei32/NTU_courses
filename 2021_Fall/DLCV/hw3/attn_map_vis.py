from pytorch_pretrained_vit import ViT
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os

model_path = './p1_model.pth'
num_classes = 37

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# Load model
model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# transforms

tfm_to_tensor = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

tfm_to_img = transforms.Compose([transforms.ToPILImage(),
                                 transforms.Resize((384,384))])


# Read 3 images
index = ['26_5064', '29_4718', '31_4838']
data_dir = "./hw3_data/p1_data/val"
imgs = []
for idx in index:
    path = os.path.join(data_dir, idx+'.jpg')
    img = Image.open(path).convert('RGB')
    imgs.append(img)

for idx, img in zip(index, imgs):  
    x = tfm_to_tensor(img)
    out = model(x.unsqueeze(0))
    scores = model.transformer.blocks[-1].attn.scores # size=(1,12,577,577), 12 heads
    
    # average across 12 heads
    scores = torch.mean(scores, dim=1) 

    # attention map between the [class] token (as query vector) and all patches (as key vectors)
    # extract row 0 (size = (1, 577)) and transpose to size=(1, 24, 24) 
    attn_map = scores.squeeze(0)[0, 1:].view(1, 24, 24)

    # map pixel values to [0, 255]
    attn_map *= 255
    attn_map = tfm_to_img(attn_map)
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)
    fig.suptitle("Visualization of Attention Map", fontsize=24)
    im1 = ax1.imshow(img.resize((384, 384)))
    im2 = ax2.imshow(attn_map)
    fig.savefig(f"attn_map_{idx}.jpg")
    