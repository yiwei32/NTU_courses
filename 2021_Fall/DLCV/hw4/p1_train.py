import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from p1_dataset import *
from p1_model import *
from utils import *

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

root_dir = "./hw4_data/mini"
model_dir = "p1_models"
os.makedirs(model_dir, exist_ok=True)
data_dir = os.path.join(root_dir, 'train')
csv_path = os.path.join(root_dir, 'train.csv')


workers = 3
lr = 1e-3
weight_decay = 0
num_epochs = 200
num_episodes = 100

# Create dataset
train_way = 10
N_shot = 1
N_query = 15

train_set = MiniDataset(csv_path, data_dir)
train_sampler = CategoriesSampler(train_set.labels, train_way, N_shot+N_query, episodes=num_episodes)
train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler,
                              num_workers=workers, worker_init_fn=worker_init_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convnet()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if __name__=='__main__':
    
    model.train()
    for epoch in range(num_epochs):
        
        for i, batch in enumerate(train_loader, 1):
            data, _ = batch
            data = data.to(device)
            p = N_shot * train_way
            data_support, data_query = data[:p], data[p:]
            proto = model(data_support)
            proto = proto.reshape(N_shot, train_way, -1).mean(dim=0)

            label = torch.arange(train_way).repeat(N_query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                    .format(epoch+1, i, len(train_loader), loss.item(), acc))

            torch.save(model.state_dict(), os.path.join(model_dir, 'p1_model.pth'))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None
        lr_scheduler.step()
