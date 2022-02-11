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


test_episodes = 600
N_way = 5
K_shot = 1
N_query = 15

class Parametric(nn.Module):
    def __init__(self):
        super(Parametric, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1600*5, 5),
            # nn.BatchNorm1d(512),
            # nn.Linear(512, 5),
        )

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
parametric_model = Parametric()

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
    # Evaluation
    for name, metric in metric_dict.items():
        print(f"start evaluating with {name} metric setting")
        model_path = os.path.join(model_dir, f'p1_2_{name}_model.pth')
        model = Convnet()
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        if name == 'param':
            parametric_model.load_state_dict(torch.load("parametric_model.pth"))
            parametric_model = parametric_model.to(device)
            parametric_model.eval()
        
        test_set = MiniDataset(test_csv_path, test_data_dir)
        test_sampler = CategoriesSampler(test_set.labels, N_way, K_shot+N_query, episodes=test_episodes)
        test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler)
        accs = []
        for i, batch in enumerate(test_loader, 1):
            with torch.no_grad():
                data, _ = batch
                data = data.to(device)
                p = K_shot * N_way
                data_support, data_query = data[:p], data[p:]
                proto = model(data_support)
                proto = proto.reshape(K_shot, N_way, -1).mean(dim=0)
                label = torch.arange(N_way).repeat(N_query)
                label = label.type(torch.LongTensor).to(device)
                logits = metric(model(data_query), proto)
                acc = count_acc(logits, label)
                accs.append(acc)
                proto = None; logits = None
        print("Acc = {:.2f} %, under {} metric setting".format(np.mean(accs)*100, name))