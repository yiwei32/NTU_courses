import torch
import torch.nn as nn
from torchvision.models import vgg16


class FCN32s(nn.Module):
    def __init__(self):
        super(FCN32s, self).__init__()
        
        self.vgg = vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            # 7 possible class labels for each pixel
            nn.Conv2d(4096, 7, kernel_size=1, stride=1),
            nn.ConvTranspose2d(7, 7, kernel_size=64, stride=32),
        )
    
    def forward (self, x) :        
        x = self.vgg.features(x)
        x = self.vgg.classifier(x)
        return x


class FCN8s(nn.Module):
    def __init__(self):
        super(FCN8s, self).__init__()
        
        self.vgg = vgg16(pretrained=True)
        self.to_pool3 = nn.Sequential(*list(self.vgg.features.children())[:17]) 
        self.to_pool4 = nn.Sequential(*list(self.vgg.features.children())[17:24])
        self.to_pool5 = nn.Sequential(*list(self.vgg.features.children())[24:])
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 7, kernel_size=1, stride=1),
            nn.ConvTranspose2d(7, 256, kernel_size=8, stride=4)
            )
        self.pool4_upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample8 = nn.ConvTranspose2d(256, 7, kernel_size=8, stride=8)
    
    def forward(self, x):
        pool3_out = self.to_pool3(x) # torch.Size([16, 256, 64, 64]) 
        # print(f"size of pool3_out= {pool3_out.shape}")
        pool4_out = self.to_pool4(pool3_out) # torch.Size([16, 512, 32, 32]) 
        # print(f"size of pool4_out= {pool4_out.shape}")  
        pool4_2x = self.pool4_upsample2(pool4_out)  # torch.Size([16, 256, 64, 64])
        # print(f"size of pool4_2x= {pool4_2x.shape}")  
        x = self.to_pool5(pool4_out)
        x = self.vgg.classifier(x) # torch.Size([16, 256, 64, 64])
        # print(f"size of classifier out = {x.shape}")  
        x = self.upsample8(x + pool3_out + pool4_2x)
        return x
