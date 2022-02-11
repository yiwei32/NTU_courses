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
train_data_dir = os.path.join(root_dir, 'train')
train_csv_path = os.path.join(root_dir, 'train.csv')
test_data_dir = os.path.join(root_dir, 'val')
test_csv_path = os.path.join(root_dir, 'val.csv')

workers = 3
lr = 1e-3
weight_decay = 0
num_epochs = 80
num_episodes = 100
test_episodes = 600

# Create dataset
N_way = 5
N_shots = [1, 5 ,10]
N_query = 15

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    for K_shot in N_shots:
        train_set = MiniDataset(train_csv_path, train_data_dir)
        train_sampler = CategoriesSampler(train_set.labels, N_way, K_shot+N_query, episodes=num_episodes)
        train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler,
                                    num_workers=workers, worker_init_fn=worker_init_fn)
    
        model_path = os.path.join(model_dir, f'p1_3_{K_shot}shot_model.pth')
        model = Convnet()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        model.train()
        print(f"start training with {K_shot} shot setting")
        for epoch in range(num_epochs):
            for i, batch in enumerate(train_loader, 1):
                data, _ = batch
                data = data.to(device)
                p = K_shot * N_way
                data_support, data_query = data[:p], data[p:]
                proto = model(data_support)
                proto = proto.reshape(K_shot, N_way, -1).mean(dim=0)

                label = torch.arange(N_way).repeat(N_query)
                label = label.type(torch.LongTensor).to(device)

                logits = euclidean_metric(model(data_query), proto)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                        .format(epoch+1, i, len(train_loader), loss.item(), acc))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                proto = None; logits = None; loss = None
            lr_scheduler.step()
            torch.save(model.state_dict(), model_path)
        print(f"{model_path} saved.")
        
