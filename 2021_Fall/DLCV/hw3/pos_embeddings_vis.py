import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pytorch_pretrained_vit import ViT

model_path = './p1_model.pth'
num_classes = 37

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# size = (1, 577, 768)
s = model.positional_embedding.pos_embedding.shape

# remove first patch
# size = (576, 768)
pos_patch = model.positional_embedding.pos_embedding.view(*s[1:])[:-1].view(24*24,-1) 

# Visualize position embedding similarities.
# One cell shows cos similarity between an embedding and all the other embeddings.
pos_embed = model.positional_embedding.pos_embedding[0,1:] #size=(576, 768)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig, axes = plt.subplots(figsize=(12,10), nrows=24, ncols=24)
fig.suptitle("Visualization of position embedding similarities", fontsize=24)

for i, ax in enumerate(axes.flatten()):
    sim = F.cosine_similarity(pos_embed[i:i+1], pos_embed[:], dim=1)
    sim = sim.reshape((24, 24)).detach().cpu().numpy()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    im = ax.imshow(sim)

plt.xlabel('Input Patch Column')
plt.ylabel('Input Patch Row')
cax,kw = mpl.colorbar.make_axes([ax for ax in fig.axes])
cbar = plt.colorbar(im, cax=cax, **kw)
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(["-1", "0", "1"])
plt.savefig("./pos_embed_vis.jpg")