import numpy as np
import dgl
import torch
import os
from dgl.data import DGLDataset
from torch.utils.data import Dataset
import random
from ultis import *
import pickle

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
    def __init__(self):
        super().__init__(name='IEMOCAP_DGL')
        self.process()


    def extractNode(self, x1, x2, x3, x4):
        text = np.asarray(x1)
        audio = np.asarray(x2)
        video = np.asarray(x3)
        speakers = torch.FloatTensor([[1]*5 if x=='M' else [0]*5 for x in x4])
        output = np.hstack([text, audio, video, speakers])
        return output    


    def extractEdge(self, datas, nodeStart):
        numUtterance = len(datas)
        x1, x2, x3 = [], [], []
        for ii in range(numUtterance - 1):
            sim = featureSimilarity(datas[ii], datas[ii+1])
            x1.append(sim)
            x2.append(nodeStart+ii)
            x3.append(nodeStart+ii+1)
            # x2.append(nodeStart+ii+1)
            # x3.append(nodeStart+ii)
            # x1.append(sim)
        sim = featureSimilarity(datas[0], datas[-1])
        x1.append(sim)
        x2.append(nodeStart)
        x3.append(nodeStart+numUtterance-1)
        for ii in range(numUtterance - 1):
            for jj in range(ii+1, numUtterance):
                sim = featureSimilarity(datas[ii], datas[jj])
                alpha = 1.0
                x1.append(alpha * sim)
                x2.append(nodeStart+ii)
                x3.append(nodeStart+jj)
                x2.append(nodeStart+jj)
                x3.append(nodeStart+ii)
                x1.append(alpha * sim)
        return x1, x2, x3

    def process(self):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')
        numSubGraph = len(self.trainVid) + len(self.testVid)
        numNodeTrain = sum([len(self.videoText[x]) for x in self.trainVid])
        numNodeTest = sum([len(self.videoText[x]) for x in self.testVid])
        numberNode = numNodeTest + numNodeTrain
        node_featuresTrain = np.vstack([self.extractNode(self.videoText[x], self.videoVisual[x], \
            self.videoAudio[x], self.videoSpeakers[x]) for x in self.trainVid])
        node_featuresTest = np.vstack([self.extractNode(self.videoText[x], self.videoVisual[x], \
            self.videoAudio[x], self.videoSpeakers[x]) for x in self.testVid])
        node_features = np.vstack([node_featuresTrain, node_featuresTest])
        # feature normalization
        node_features = norm(node_features)
        node_labelTrain = np.hstack([np.asarray(self.videoLabels[x]) for x in self.trainVid])
        node_labelTest = np.hstack([np.asarray(self.videoLabels[x]) for x in self.testVid])
        node_labels = np.hstack([node_labelTrain, node_labelTest])
        self.num_classes = np.unique(node_labels)

        node_features =  torch.from_numpy(node_features).double()
        node_labels =  torch.from_numpy(node_labels).long()
        edge_features = []
        edges_src = []
        edges_dst = []
        counter = 0
        for dataset in [self.trainVid, self.testVid] :
            for idx, x in enumerate(dataset):
                numUtterance = len(self.videoLabels[x])
                x1, x2, x3 = self.extractEdge(node_features[counter: counter+numUtterance], counter)
                edge_features.append(x1)
                edges_src.append(x2)
                edges_dst.append(x3)
                counter += numUtterance

        edge_features, edges_src, edges_dst = convertNP2Tensor([ np.hstack(edge_features), np.hstack(edges_src), np.hstack(edges_dst)])
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=numberNode)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        self.n_nodes = numberNode
        n_nodes = self.n_nodes
        n_train = numNodeTrain
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        test_mask[n_train:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


    def genMissing(self, typeMissing = 10):
        mm = genMissMultiModal((3, 1, 10), 2)
        mm = np.asarray(mm[0])
        print(f'missing percent: {mm.sum()*1.0 / (mm.shape[0] * mm.shape[1])}')
        missingGraph = None
        return missingGraph


    def reconstruct(self, typeReconstruct):
        pass

# trainset = IEMOCAPDataset()
# print(trainset[1][0].shape)

# trainsetDGL = IEMOCAP()
