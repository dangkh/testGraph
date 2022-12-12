import random 
import torch 
import numpy as np 
from torch import nn

seed = 27350

def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
def featureSimilarity(v1, v2):
    similar = torch.acos(cos(v1, v2))/ np.pi
    return similar

def convertNP2Tensor(listV):
    listR = []
    for xx in listV:
        listR.append(torch.from_numpy(xx))
    return listR