import sys
import os
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from p1_dataset import p1_data
from torchvision.models import resnet152

# create testing dataset
model_path = sys.argv[1]
in_dir = sys.argv[2]
out_path = sys.argv[3]
batch_size = 32

test_tfm = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

test_set = p1_data(in_dir, 'test', transform=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# create model and load weights from checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet152(pretrained=True)
# modify the last layer
model.fc = nn.Linear(2048, 50)

model.load_state_dict(torch.load(model_path))
model = model.to(device)

predict = []
filenames = []
model.eval() # set the model to evaluation mode

for batch in test_loader:
    # create fake labels to make it work normally
    imgs, labels, fns = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    predict.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    filenames.extend(fns)
with open(out_path, 'w') as f:
    f.write('image_id,label\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(filenames[i],y))
