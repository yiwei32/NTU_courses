import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms  
from PIL import Image  


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

image_size = 28
workers = 0
num_label = 10
num_per_class = 100
tfm =transforms.Compose([
        transforms.Resize(image_size),
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
        self.filenames = sorted([file for file in os.listdir(root) if file.endswith('.png')])
        
        #read labels
        labels = np.array([x * np.ones(num_per_class) for x in range(num_label)], dtype=np.int32).flatten()
        self.labels = torch.Tensor(labels)
        
        self.len = len(labels)
    def __getitem__(self, index):
        filepath = os.path.join(self.root, self.filenames[index])
        image = Image.open(filepath)
        image = self.transform(image)
        
        return image, self.labels[index]
    def __len__(self):
        return self.len
        
                        


if __name__ == '__main__':
    
    # load digit classifier
    net = Classifier()
    path = "Classifier.pth"
    load_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    print(net)
    workers = 0
    batch_size = 1
    dataroot = './p2_output'
    dataset = mnistmDataset(root=dataroot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)   
    test_accs = []
    for i, data in enumerate(dataloader):
        imgs, labels = data
        with torch.no_grad():
          logits = net(imgs.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        test_accs.append(acc)
    
    test_acc = sum(test_accs) / len(test_accs)


    print(f"acc = {test_acc:.5f}")



