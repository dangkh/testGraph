import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.nn import LabelPropagation
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from loadData import *
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from AttentionInnerModule import *
# torch.set_default_dtype(torch.float)
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


def check(data):
	if len(np.where(data==0)[0]) > 0:
		return True
	return False

path = 'F:/dangkh/work/dgl/graphIEMOCAP/missing_30_test4.dgl'
(g,), _ = dgl.load_graphs(path)
features = g.ndata["feat"]
sourceFile = open('debug2.txt', 'w')

lenF = features.shape[0]
for ii in range(lenF):
	tmp = [0,0,0]
	line = features[ii]
	te = line[:100]
	au = line[100:442]
	vi = line[442:]
	for ind, i in enumerate([te,au,vi]):
		if check(i):
			tmp[ind] = 1
	print(tmp, file = sourceFile)

sourceFile.close()
