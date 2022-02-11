import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from PIL import Image
import sys

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
model_path = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

output_size = 28

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(output_size)])

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 80

# Size of feature maps in discriminator
ndf = 80

# Number of classes
num_label = 10
                        
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz=nz, ngf=ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
            )
    
    def forward(self, input):
        return self.main(input)

# Load model
netG = Generator(nz)
netG.load_state_dict(torch.load(model_path))
netG.eval()
netG.to(device)

# Create 1000 latent vectors and embed label inside
num_sample = 1000
num_per_class = 100
fixed_noise = torch.FloatTensor(num_sample, nz, 1, 1).normal_(0, 1)
fixed_noise = Variable(fixed_noise).to(device)
fixed_noise_ = np.random.normal(0, 1, (num_sample, nz))
label = np.array([x * np.ones(num_per_class) for x in range(num_label)], dtype=np.int32).flatten()
label_onehot = np.zeros((num_sample, num_label))
label_onehot[np.arange(num_sample), label] = 1
fixed_noise_[np.arange(num_sample), :num_label] = label_onehot[np.arange(num_sample)]

fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(num_sample, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_)

# Generate 1000 images and save them.
imgs_sample = (netG(fixed_noise).data + 1) / 2.0 # denormalize
for idx, img in enumerate(imgs_sample, 0):
        img_label = idx // num_per_class
        img_idx = idx % num_per_class
        number = str(img_idx+1).zfill(3)
        img = tfm(img)
        img.save(os.path.join(output_dir, f'{img_label}_{number}.png'))
print("1000 images generated!")
