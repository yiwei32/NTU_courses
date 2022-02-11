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
K_shot = 1
N_query = 15

train_set = MiniDataset(train_csv_path, train_data_dir)
train_sampler = CategoriesSampler(train_set.labels, N_way, K_shot+N_query, episodes=num_episodes)
train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler,
                              num_workers=workers, worker_init_fn=worker_init_fn)

class Parametric(nn.Module):
    def __init__(self):
        super(Parametric, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1600*5, 5),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Linear(512, 5),
        )

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))
    
## takes in a module and applies the specified weight initialization
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)
        
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
parametric_model = Parametric()
parametric_model.apply(weights_init_normal)
parametric_model = parametric_model.to(device)
parametric_optimizer = torch.optim.Adam(parametric_model.parameters(), lr=1e-5, weight_decay=weight_decay)


def parametric_function(query, proto):
    n = query.shape[0] 
    m = proto.shape[0] 
    # size of query = (query, 1600) -> (1, batch, 1600) repeat-> (N_way, query, 1600)
    # size of proto = (N_way, 1600) -> (N_way, 1, 1600) repeat-> (N_way, query, 1600)
    query = query.unsqueeze(1).expand(n, m, -1) 
    proto = proto.unsqueeze(0).expand(n, m, -1) 
    
    return parametric_model(query-proto)

if __name__=='__main__':
    metric_dict = {'euc' : euclidean_metric, 'cos' : cos_sim_metric, 'param' : parametric_function}
    for name, metric in metric_dict.items():
        model_path = os.path.join(model_dir, f'p1_2_{name}_model.pth')
        model = Convnet()
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        model.train()
        parametric_model.train()
        print(f"start training with metric function {name}")
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

                logits = metric(model(data_query), proto)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                        .format(epoch+1, i, len(train_loader), loss.item(), acc))

                optimizer.zero_grad()
                
                if name == 'param':
                    parametric_optimizer.zero_grad()
                    loss.backward()
                    parametric_optimizer.step()
                else:
                    loss.backward()
                
                optimizer.step()
                
                proto = None; logits = None; loss = None
            lr_scheduler.step()
            torch.save(model.state_dict(), model_path)
        print(f"{model_path} saved.")
        if name == 'param':
            torch.save(parametric_model.state_dict(), "parametric_model.pth")
