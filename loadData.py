import numpy as np
import dgl
import torch
import os
from dgl.data import DGLDataset
from torch.utils.data import Dataset
import random
from ultis import *
import pickle


def missingParam(percent):
    al, be , ga = 0, 0, 0
    for aa in range(1, 200):
        for bb in range(1, 200):
            for gg in range(200):
                if (aa+bb+gg) != 0:
                    if abs(((bb*3 + gg * 6) * 100.0 / (aa*3 + bb*9 + gg*6)) - percent) <= 1.0:
                        return aa, bb, gg
    return al, be, ga


def genMissMultiModal(matSize, percent):
    index = (percent-10) // 10
    types = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    missPercent = 0
    batch_size = 1
    if matSize[0] != len(types[0]):
        return None
    al, be, ga = missingParam(percent)
    errPecent = 1.7
    if matSize[-1] <= 5:
        errPecent = 5
    if matSize[-1] <= 2:
        errPecent = 20
    listMask = []
    masks = [np.asarray([[0, 0, 0]]), np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), np.asarray([[0, 1, 1], [1, 1, 0], [1, 0, 1]])]
    for mask, num in ([0, al], [1, be], [2, ga]):
        if num > 0:
            listMask.append(np.repeat(masks[mask], num, axis = 0))
    missType = np.vstack(listMask)
    counter = 0
    while np.abs(missPercent - percent) > 1.0:
        mat = np.zeros((matSize[0], matSize[-1]))
        for ii in range(matSize[-1]):
            tmp = random.randint(0, len(missType)-1)
            mat[:,ii] = missType[tmp]
        missPercent = mat.sum() / (matSize[0] * matSize[-1]) * 100
        if (np.abs(missPercent - percent) < errPecent) & (np.abs(missPercent - percent) > 0):
            return mat
    return np.zeros((matSize[0], matSize[-1]))

class IEMOCAP(DGLDataset):
    def __init__(self, nameDataset='IEMOCAP', path = './IEMOCAP_features/IEMOCAP_features.pkl',\
            mergeLabel = False, missing = 0, edgeType = 0, batchSize = 16):
        self.missing = missing
        self.edgeType = edgeType
        self.path = path
        self.dataset = nameDataset
        self.mergeLabel = mergeLabel
        self.batchSize = batchSize
        super().__init__(name='dataset_DGL')


    def extractNode(self, x1, x2, x3, x4):
        text = np.asarray(x1)
        audio = np.asarray(x2)
        video = np.asarray(x3)
        speakers = torch.FloatTensor([[1]*5 if x=='M' else [0]*5 for x in x4])
        # 100, 342, 1582, 5
        output = np.hstack([text, audio, video, speakers])
        return output    


    def extractEdge(self, datas, nodeStart):
        numUtterance = len(datas)
        if self.missing > 0:
            self.missingMask = genMissMultiModal((3, numUtterance), self.missing)
            for ii in range(numUtterance):
                currentFeatures = datas[ii]
                text = currentFeatures[:100]
                audio = currentFeatures[100: 442]
                video = currentFeatures[442: 442+1582]
                if self.missingMask[0][ii] == 1:
                    text[:] = 0
                if self.missingMask[1][ii] == 1:
                    audio[:] = 0
                if self.missingMask[2][ii] == 1:
                    video[:] = 0
        x1, x2, x3 = [], [], []
        for ii in range(numUtterance - 1):
            sim = 1
            if self.edgeType == 0:
                sim = featureSimilarity(datas[ii], datas[ii+1])
            x1.append(sim)
            x2.append(nodeStart+ii)
            x3.append(nodeStart+ii+1)

        sim = featureSimilarity(datas[0], datas[-1])
        x1.append(sim)
        x2.append(nodeStart)
        x3.append(nodeStart+numUtterance-1)
        for ii in range(numUtterance - 1):
            for jj in range(ii+1, numUtterance):
                sim = 1
                if self.edgeType == 0:
                    sim = featureSimilarity(datas[ii], datas[jj])
                x1.append(sim)
                x2.append(nodeStart+ii)
                x3.append(nodeStart+jj)
                x2.append(nodeStart+jj)
                x3.append(nodeStart+ii)
                x1.append(sim)
        return x1, x2, x3

    def process(self):
        self.subIdTrain, self.subIdTest = [], []
        inputData = pickle.load(open(self.path, 'rb'), encoding='latin1')
        if self.dataset == 'MELD':
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
            self.testVid, _ = inputData
        else:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
            self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
            self.testVid = inputData
        
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
        if self.mergeLabel:
            node_labels[np.where(node_labels == 4)] = 0
            node_labels[np.where(node_labels == 5)] = 1
            if self.dataset == 'MELD':
                node_labels[np.where(node_labels == 6)] = 4
        self.num_classes = np.unique(node_labels)
        node_features =  torch.from_numpy(node_features).double()
        node_labels =  torch.from_numpy(node_labels).long()
        edge_features = []
        edges_src = []
        edges_dst = []
        counter = 0
        for idx, x in enumerate(self.trainVid):
            numUtterance = len(self.videoLabels[x])
            x1, x2, x3 = self.extractEdge(node_features[counter: counter+numUtterance], counter)
            edge_features.append(x1)
            edges_src.append(x2)
            edges_dst.append(x3)
            self.subIdTrain.append(np.unique(np.asarray(np.hstack([x2, x3]))).reshape(1,-1))
            counter += numUtterance
        for idx, x in enumerate(self.testVid):
            numUtterance = len(self.videoLabels[x])
            x1, x2, x3 = self.extractEdge(node_features[counter: counter+numUtterance], counter)
            edge_features.append(x1)
            edges_src.append(x2)
            edges_dst.append(x3)
            self.subIdTest.append(np.unique(np.asarray(np.hstack([x2, x3]))).reshape(1,-1))
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
        self.batched_graph(self.batchSize)

    def __getitem__(self, i):
        return self.graph, self.train, self.test

    def __len__(self):
        return 1

    def batched_graph(self, batchSize):
        self.listGraphTrain = []
        batchedTrain, batchedTest = None, None
        startId, rangeSample = 0, batchSize
        while rangeSample < len(self.trainVid):
            ids = self.subIdTrain[startId: rangeSample]
            ids = np.unique(np.hstack(ids))
            startId, rangeSample = startId+ batchSize, rangeSample+batchSize
            self.listGraphTrain.append(self.graph.subgraph(ids))
        self.listGraphTest = []
        batchedTrain, batchedTest = None, None
        startId, rangeSample = 0, batchSize
        while rangeSample < len(self.testVid):
            ids = self.subIdTest[startId: rangeSample]
            ids = np.unique(ids[0])
            startId, rangeSample = startId+ batchSize, rangeSample+batchSize
            self.listGraphTest.append(self.graph.subgraph(ids))
        self.train, self.test = dgl.batch(self.listGraphTrain), dgl.batch(self.listGraphTest)
        return self.train, self.test

# trainset = IEMOCAPDataset()
# print(trainset[1][0].shape)

# trainsetDGL = IEMOCAP()
# a,b,c = trainsetDGL[0]
# dgl.save_graphs('graphPath', a)
# dgl.save_graphs('graphPath1', b)
# dgl.save_graphs('graphPath2', c)
# sgTrain, sgTest = trainsetDGL.batched_graph(16)
# for ii in range(10, 70, 10):
#     genMissMultiModal((3, 8), ii)