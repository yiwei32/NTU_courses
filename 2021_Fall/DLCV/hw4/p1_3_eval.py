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
test_episodes = 600

# Create dataset
N_way = 5
N_shots = [1, 5 ,10]
N_query = 15

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__=='__main__':
    # Evaluation
    for K_shot in N_shots:
        print(f"start evaluating with {K_shot} shot setting")
        model_path = os.path.join(model_dir, f'p1_3_{K_shot}shot_model.pth')
        model = Convnet()
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
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
                logits = euclidean_metric(model(data_query), proto)
                acc = count_acc(logits, label)
                accs.append(acc)
                proto = None; logits = None
        print("Acc = {:.2f} %, under {} way {} shot setting".format(np.mean(accs)*100, N_way, K_shot))