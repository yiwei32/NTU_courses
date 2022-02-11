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

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
workspace_dir = '.'
dataroot = "./hw2_data/digits/mnistm"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28
input_size = 64

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

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

log_dir = os.path.join(workspace_dir, 'p2_logs')
ckpt_dir = os.path.join(workspace_dir, 'p2_checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

tfm = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class mnistmDataset(Dataset):
    def __init__(self, root, transform=tfm):
        """ Initialize the dataset """
        self.root = root
        self.labels = []
        self.filenames = []
        self.transform = transform
        
        
        # read images names
        train_dir = os.path.join(root, "train")
        self.filenames = sorted([file for file in os.listdir(train_dir) if file.endswith('.png')])
        
        #read labels
        labelpath = os.path.join(dataroot, 'train.csv')
        labels = pd.read_csv(labelpath).iloc[:, 1]
        self.labels = torch.Tensor(labels)
        
        self.len = len(labels)
    def __getitem__(self, index):
        train_dir = os.path.join(self.root, "train")
        filepath = os.path.join(train_dir, self.filenames[index])
        image = Image.open(filepath)
        image = self.transform(image)
        
        return image, self.labels[index]
    
    def __len__(self):
        return self.len
        
                        
dataset = mnistmDataset(root=dataroot)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
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
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
            )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf=ndf, num_label=num_label):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0, bias=False),
            )
        self.dis_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, num_label)
        self.ndf = ndf
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)
    
    
    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)
        # c = self.softmax(c)
        s = self.dis_linear(x)
        s = self.sigmoid(s)
        
        return s, c

# Create the generator and discriminator
netG = Generator().to(device)
netD = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)
netD.apply(weights_init)

# Print the model
print(netG)
print(netD)

# Initialize Loss function
dis_criterion = nn.BCELoss()
aux_criterion = nn.CrossEntropyLoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
num_sample = 32
fixed_noise = torch.FloatTensor(num_sample, nz, 1, 1).normal_(0, 1)
fixed_noise = Variable(fixed_noise).to(device)
fixed_noise_ = np.random.normal(0, 1, (num_sample, nz))
random_label = np.random.randint(0, num_label, num_sample)
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((num_sample, num_label))
random_onehot[np.arange(num_sample), random_label] = 1
fixed_noise_[np.arange(num_sample), :num_label] = random_onehot[np.arange(num_sample)]

fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(num_sample, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_)


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-4)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-4)


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

def test(predict, labels):
    softmax = nn.Softmax(dim=-1)
    predict = softmax(predict)
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        
        ###########################
        # (1) Update D network
        ###########################
        # train with real
        netD.zero_grad()
        img, label = data
        img = img.to(device)
        b_size = img.size(0)
        s_label = Variable(torch.FloatTensor(b_size, ).uniform_(0.7, 1)).to(device)
        c_label = Variable(label.long()).to(device)
        s_output, c_output = netD(img)
        s_output = s_output.view(-1)
        s_errD_real = dis_criterion(s_output, s_label)
        c_errD_real = aux_criterion(c_output, c_label)
        errD_real = s_errD_real + c_errD_real
        errD_real.backward()
        D_x = s_output.data.mean()
        real_correct, real_length = test(c_output, c_label)
        
        # train with fake
        noise = np.random.randn(b_size, nz)
        label = np.random.randint(0, num_label, b_size) # random label vector length = b_size
   
        label_onehot = np.zeros((b_size, num_label)) # (128, 10)
        label_onehot[np.arange(b_size), label] = 1
        # put label_onehot inside the noise
        noise[np.arange(b_size), :num_label] = label_onehot[np.arange(b_size)]
        
        noise = Variable(torch.FloatTensor(noise).reshape(b_size, nz, 1, 1)).to(device)
        
        c_label = Variable(torch.from_numpy(label).long()).to(device)
        
        fake = netG(noise)
        s_label = Variable(torch.FloatTensor(b_size, ).uniform_(0., 0.3)).to(device)
        s_output, c_output = netD(fake.detach())
        s_output = s_output.view(-1)
        s_errD_fake = dis_criterion(s_output, s_label)
        c_errD_fake = aux_criterion(c_output, c_label)
        errD_fake = s_errD_fake + c_errD_fake
        
        errD_fake.backward()
        D_G_z1 = s_output.data.mean()
        errD = s_errD_real + s_errD_fake

        fake_correct, fake_length = test(c_output, c_label)
        optimizerD.step()
        
        ###########################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        # fake labels are real for generator cost
        s_label = Variable(torch.FloatTensor(b_size, ).uniform_(0.7, 1)).to(device)
        s_output, c_output = netD(fake)
        s_output = s_output.view(-1)
        s_errG = dis_criterion(s_output, s_label)
        c_errG = aux_criterion(c_output, c_label)
        
        errG = s_errG + c_errG
        errG.backward()
        D_G_z2 = s_output.data.mean()
        optimizerG.step()
        
        # Output training stats
        if i % 50 == 0: 
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, real_Acc: %d / %d = %.2f, fake_Acc: %d / %d = %.2f'
              % (epoch+1, num_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                 real_correct, real_length, 100. * real_correct / real_length, fake_correct, fake_length, 100 * fake_correct / fake_length))
        
            
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
            
        iters += 1


    # Save checkpoints
    torch.save(netG.state_dict(), os.path.join(ckpt_dir, f'p2_netG_{epoch+1:03d}.pth'))
    torch.save(netD.state_dict(), os.path.join(ckpt_dir, f'p2_netD_{epoch+1:03d}.pth'))

    netG.eval()
    f_imgs_sample = (netG(fixed_noise).data + 1) / 2.0
    filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
    vutils.save_image(f_imgs_sample, filename, nrow=8)
    print(f' | Save some samples to {filename}.')
    netG.train()


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./p2_logs/Loss.png")

