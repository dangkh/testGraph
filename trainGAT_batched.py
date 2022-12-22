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
# torch.set_default_dtype(torch.float)
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        gcv = [in_size, 512, 32]
        self.num_heads = 4
        self.layers = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.layers.append(
                dglnn.GATv2Conv(np.power(self.num_heads, ii) * gcv[ii],  gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        self.linear = nn.Linear(gcv[-1] * self.num_heads, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = h.float()
            h = torch.reshape(h, (len(h), -1))
            h = layer(g, h)
        h = torch.reshape(h, (len(h), -1))
        h = self.linear(h)
        return h

class GAT_FP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, numFP):
        super().__init__()
        gcv = [in_size, 512, 32]
        self.num_heads = 4
        self.layers = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.layers.append(
                dglnn.GATv2Conv(np.power(self.num_heads, ii) * gcv[ii],  gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        # self.layers.append(dglnn.GraphConv(hid_size, 16))
        self.linear = nn.Linear(gcv[-1] * self.num_heads, out_size)
        self.dropout = nn.Dropout(0.5)
        self.label_propagation = LabelPropagation(k=numFP, alpha=0.5, clamp=False, normalize=True)

    def forward(self, g, features):
        h = features
        h = self.label_propagation(g, features)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = h.float()
            h = torch.reshape(h, (len(h), -1))
            h = layer(g, h)
        h = torch.reshape(h, (len(h), -1))
        h = self.linear(h)
        return h


def train(g, trainSet, testSet, masks, model, info):
    # define train/val samples, loss function and optimizer
    if info['MSE'] == False:
        loss_fcn = nn.CrossEntropyLoss()
        # loss_fcn = FocalLoss()
    else:
        loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=info['lr'], weight_decay=info['weight_decay'])
    highestAcc = 0
    # training loop
    listTrain = dgl.unbatch(trainSet)
    for epoch in range(info['numEpoch']):
        totalLoss = 0
        for graph in listTrain:
            features = graph.ndata["feat"]
            labels = graph.ndata["label"]
            model.train()
            logits = model(graph, features)
            loss = loss_fcn(logits, labels)
            totalLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(g, masks[0], model)
        acctest = evaluate(g, masks[1], model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy_train {:.4f} | Accuracy_test {:.4f} ".format(
                epoch, totalLoss, acc, acctest
            )
        )
        highestAcc = max(highestAcc, acctest)

    return highestAcc
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--numEpoch', help='number of epochs', default=50, type=int)
    parser.add_argument('--seed', help='type of seed: random vs fix', default='random')
    parser.add_argument('--lr', help='learning rate', default=0.003, type=float)
    parser.add_argument('--weight_decay', help='weight decay', default=0.00001, type=float)
    parser.add_argument('--edgeType', help='type of edge:0 for similarity and 1 for other', default=0, type=int)
    parser.add_argument('--missing', help='percentage of missing utterance in MM data', default=10, type=int)
    parser.add_argument('--wFP', action='store_true', default=False, help='edge direction type')
    parser.add_argument('--numFP', help='number of FP layer', default=5, type=int)
    parser.add_argument('--numTest', help='number of test', default=10, type=int)
    parser.add_argument('--batchSize', help='size of batch', default=8, type=int)
    parser.add_argument('--mergeLabel', help='if True then mergeLabel from 6 to 4',action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=True, help='save experiment info in output')
    parser.add_argument('--output', help='savedFile', default='./log.txt')
    parser.add_argument('--MSE', help='reduce variant in laten space',  action='store_true', default=False)
    parser.add_argument( "--dataset",
        type=str,
        default="IEMOCAP",
        help="Dataset name ('IEMOCAP', 'MELD').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")
    torch.cuda.empty_cache()
    info = {
            'numEpoch': args.numEpoch,
            'lr': args.lr, 
            'weight_decay': args.weight_decay,
            'missing': args.missing,
            'seed': 'random',
            'numTest': args.numTest,
            'wFP': args.wFP,
            'numFP': args.numFP,
            'MSE': args.MSE
        }

    for test in range(args.numTest):
        if args.seed == 'random':
            setSeed = random.randint(1, 100001)
            info['seed'] = setSeed
        else:
            setSeed = int(args.seed)
        seed_everything(seed=setSeed)
        if args.log:
            sourceFile = open(args.output, 'a')
            print('*'*10, 'INFO' ,'*'*10, file = sourceFile)
            print(info, file = sourceFile)
            sourceFile.close()
        graphPath = f'./graph{args.dataset}/{args.dataset}.dgl'
        trainPath = f'./graph{args.dataset}/{args.dataset}_trainset.dgl'
        testPath = f'./graph{args.dataset}/{args.dataset}_testset.dgl'
        if args.missing > 0:
            graphPath = f'./graph{args.dataset}/missing_{args.missing}_test{test}.dgl'

        print("generating MM graph")
        if os.path.isfile(graphPath):
            (g,), _ = dgl.load_graphs(graphPath)
            trainPath = f'./graph{args.dataset}/missing_{args.missing}_test{test}_trainset.dgl'
            (trainSet,), _ = dgl.load_graphs(trainPath)
            testPath = f'./graph{args.dataset}/missing_{args.missing}_test{test}_testset.dgl'
            (testSet,), _ = dgl.load_graphs(testPath)
        else:        
            data = IEMOCAP()
            g, trainSet, testSet = data[0]
            dgl.save_graphs(graphPath, g)
            trainPath = f'./graph{args.dataset}/missing_{args.missing}_test{test}_trainset.dgl'
            dgl.save_graphs(trainPath, trainSet)
            testPath = f'./graph{args.dataset}/missing_{args.missing}_test{test}_testset.dgl'
            dgl.save_graphs(testPath, testSet)
        print("loaded MM graph")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')
        g = g.to(device)
        trainSet = trainSet.to(device)
        testSet = testSet.to(device)
        features = g.ndata["feat"]
        labels = g.ndata["label"]
        if args.MSE:
            labels = convertX2Binary(labels)
        masks = g.ndata["train_mask"], g.ndata["test_mask"]

        # create GCN model
        in_size = features.shape[1]
        out_size = torch.unique(g.ndata['label']).shape[0]
        if args.wFP:
            model = GAT_FP(in_size, 128, out_size, args.numFP).to(device)    
        else:
            model = GAT(in_size, 128, out_size).to(device)
        print(model)
        # model training
        print("Training...")
        highestAcc = train(g, trainSet, testSet, masks, model, info)

        # test the model
        print("Testing...")
        print(features.shape)
        if args.MSE:
            acc = evaluateMSE(g, testSet, labels, masks[1], model, visual=True, originalLabel=g.ndata["label"])
        else:
            labels = g.ndata["label"]
            acc = evaluate(g, masks[1], model)
        print("Final Test accuracy {:.4f}".format(acc))

        if args.log:
            sourceFile = open(args.output, 'a')
            print(f'Highest Acc: {highestAcc}, final Acc {acc}', file = sourceFile)
            print('*'*10, 'End' ,'*'*10, file = sourceFile)
            sourceFile.close()