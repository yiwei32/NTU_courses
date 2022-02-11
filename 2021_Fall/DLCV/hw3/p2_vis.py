import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

from transformers import BertTokenizer
from PIL import Image
import argparse

from saahiluppal_catr_master.models import caption
from saahiluppal_catr_master.datasets import coco, utils
from saahiluppal_catr_master.configuration import Config
import os
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--input_path', type=str, help='path to image', required=True)
parser.add_argument('--output_path', type=str, help='output dir', required=True)
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='./checkpoint.pth')
args = parser.parse_args()
image_path = args.input_path
output_path = args.output_path
checkpoint_path = args.checkpoint

os.makedirs(output_path, exist_ok=True)

config = Config()
path = '.'
torch.hub.set_dir(path)

print("Checking for checkpoint.")
if checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model,_ = caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

tfm = transforms.Compose([
    # transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class p2_data(Dataset):
    def __init__(self, root, transform=None):
        """ Initialize the dataset """
        self.root = root
        self.transform = transform
        # read filenames
        self.filenames = [file for file in os.listdir(root)]         
        self.len = len(self.filenames)
    
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        filepath = os.path.join(self.root, self.filenames[index])
        img = Image.open(filepath).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.filenames[index]

    def __len__(self):
        return self.len

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)


@torch.no_grad()
def evaluate():
    model.eval()
    attn_weights_list = []
    for i in range(config.max_position_embeddings - 1):
        predictions, attn_weights= model(image, caption, cap_mask)
        # attn_weights size = (batch, max_position_embeddings, h_patch, w_patch)
        predictions = predictions[:, i, :]
        attn_weights = attn_weights[:,i,:,:]
        predicted_id = torch.argmax(predictions, axis=-1)
        attn_weights_list.append(attn_weights)
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        if predicted_id[0] == 102:
            break
    return caption, attn_weights_list

# Create dataset

dataset = p2_data(root=image_path, transform=tfm)
nimgs = len(dataset)

for i in range(nimgs):
    image, fn = dataset[i]
    # for plotting, add denormalized img
    attn_plot = []
    attn_plot.append(image.permute(1,2,0) / 2 + 0.5) # (C, H, W) -> (H, W, C)
    image = image.unsqueeze(0)
    output, attn_weights_list = evaluate()
    h, w = attn_plot[0].shape[:2] # image size (h, w)

    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    result = "<start> " + result[:-1] + " <end> " 
    attn_weights_list.pop(-2) # remove the attention map correspoding to '.'

    for attn in attn_weights_list:
        attn_resized = F.resize(attn, (h,w)).permute(1,2,0)
        attn_map = (attn_resized - attn_resized.min()) / (attn_resized.max()- attn_resized.min())
        attn_plot.append(attn_map)

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Visualization", fontsize=24)
    nplots = len(attn_plot)
    nrows = int(np.ceil(nplots / 4))

    for i in range(nplots):
        ax = fig.add_subplot(nrows, 4, i+1)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        title, result = result.split(' ', 1)
        ax.set_title(title)
        ax.imshow(attn_plot[i])
    fn = fn.split('.', 1)[0] + '.png'
    fig.savefig(os.path.join(output_path,fn))