import torch
import torch.nn as nn
import torch.nn.functional as F

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def euclidean_metric(query, proto):
    # size of proto = (N_way, 1600)
    n = query.shape[0] 
    m = proto.shape[0] 
    # size of query = (query, 1600) -> (1, batch, 1600) repeat-> (N_way, query, 1600)
    # size of proto = (N_way, 1600) -> (N_way, 1, 1600) repeat-> (N_way, query, 1600)
    query = query.unsqueeze(1).expand(n, m, -1) 
    proto = proto.unsqueeze(0).expand(n, m, -1) 
    logits = -((query - proto)**2).sum(dim=2)
    # size = (5, 75) for 5-way setting, query = N_way * N_query = 5 * 15 = 75
    return logits

def cos_sim_metric(query, proto):
    # size of protos = (N-way, 1600)
    # size of query = (query, 1600)
    # unsqueeze query at dimension 1 for pytorch broadcasting operations
    logits = F.cosine_similarity(query.unsqueeze(1), proto, dim=-1)
    # size of logits = (N_way, query) = (5, 75)
    # Each 5*1 row vector means the cosine similarities between one query image and each proto
    return logits

