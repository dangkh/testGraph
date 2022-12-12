import numpy as np
import dgl
import torch
import os
from dgl.data import DGLDataset
import random
from ultis import *

def genMissMultiModal(matSize, index):
    types = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    typeExtend = [7, 3, 1]
    percentIndex = [10, 20, 30]
    missPercent = 0
    # batch_size = matSize[-1]
    batch_size = 1
    if matSize[0] != len(types[0]):
        return None
    if typeExtend[index] > 0:
        missType = np.vstack([types, np.asarray([[0, 0, 0]] * typeExtend[index])])
    elif typeExtend[index] == -2:
        missType = types
    elif typeExtend[index] == -3:
        missType = np.vstack([types, np.asarray([[0, 0, 0]] * 2), dtypes])
    else:
        missType = np.vstack([types, np.asarray([[0, 0, 0]])])
    listmask = []
    for batchId in range(batch_size):        
        counter = 0
        while np.abs(missPercent - percentIndex[index]) > 1.0:
            mat = np.zeros((matSize[0], matSize[-1]))
            for ii in range(matSize[-1]):
                tmp = random.randint(0, len(missType)-1)
                mat[:,ii] = missType[tmp]
            missPercent = mat.sum() / (matSize[0] * matSize[-1]) * 100
            print(missPercent)
            if np.abs(missPercent - percentIndex[index]) < 1.0:
                listmask.append(mat)
                missPercent = 0
                break
    return listmask

class IEMOCAP(DGLDataset):
    def __init__(self, args):
        super().__init__(name='IEMOCAP_DGL')
        self.args = args


    # def process(self):
    #     nodes_data = pd.read_csv('./members.csv')
    #     edges_data = pd.read_csv('./interactions.csv')
    #     node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
    #     node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
    #     edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
    #     edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
    #     edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())

    #     self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
    #     self.graph.ndata['feat'] = node_features
    #     self.graph.ndata['label'] = node_labels
    #     self.graph.edata['weight'] = edge_features

    #     # If your dataset is a node classification dataset, you will need to assign
    #     # masks indicating whether a node belongs to training, validation, and test set.
    #     n_nodes = nodes_data.shape[0]
    #     n_train = int(n_nodes * 0.6)
    #     n_val = int(n_nodes * 0.2)
    #     train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    #     val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    #     test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    #     train_mask[:n_train] = True
    #     val_mask[n_train:n_train + n_val] = True
    #     test_mask[n_train + n_val:] = True
    #     self.graph.ndata['train_mask'] = train_mask
    #     self.graph.ndata['val_mask'] = val_mask
    #     self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


    def genMissing(self, typeMissing = 10):

        return self.graph


    def reconstruct(self, typeReconstruct):
        pass

# dataset = IEMOCAP()
# graph = dataset[0]

# print(graph)

mm = genMissMultiModal((3, 1, 10), 2)
mm = np.asarray(mm[0])
print(mm.sum()*1.0 / (mm.shape[0] * mm.shape[1]))