import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.nn import LabelPropagation
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from loadIEMOCAP import *
torch.set_default_dtype(torch.float)


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        gcv = [in_size, 512, 32]
        self.num_heads = 4
        self.layers = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.layers.append(
                dglnn.GATConv(gcv[ii], gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        # self.layers.append(dglnn.GraphConv(hid_size, 16))
        self.linear = nn.Linear(gcv[-1] * np.power(self.num_heads, len(gcv)-1), out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = torch.reshape(h, (len(h), -1))
        h = self.linear(h)
        return h

class GAT_FP(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        gcv = [in_size, 512, 32]
        self.num_heads = 4
        self.layers = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.layers.append(
                dglnn.GATConv(gcv[ii], gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        # self.layers.append(dglnn.GraphConv(hid_size, 16))
        self.linear = nn.Linear(gcv[-1] * np.power(self.num_heads, len(gcv)-1), out_size)
        self.dropout = nn.Dropout(0.5)
        self.label_propagation = LabelPropagation(k=5, alpha=0.5, clamp=False, normalize=True)

    def forward(self, g, features):
        h = features
        h = self.label_propagation(g, features)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = torch.reshape(h, (len(h), -1))
        h = self.linear(h)
        return h


def train(g, features, labels, masks, model, info):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=info['lr'], weight_decay=info['weight_decay'])
    highestAcc = 0
    # training loop
    for epoch in range(info['numEpoch']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, train_mask, model)
        acctest = evaluate(g, features, labels, masks[1], model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy_train {:.4f} | Accuracy_test {:.4f} ".format(
                epoch, loss.item(), acc, acctest
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
    parser.add_argument('--reconstruct', action='store_true', default=False, help='edge direction type')
    parser.add_argument('--numTest', help='number of test', default=10, type=int)
    parser.add_argument('--log', action='store_true', default=True, help='save experiment info in log.txt')
    parser.add_argument('--output', help='savedFile', default='./result.txt')
    parser.add_argument( "--dataset",
        type=str,
        default="IEMOCAP",
        help="Dataset name ('IEMOCAP', 'MELD').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    info = {
            'numEpoch': args.numEpoch,
            'lr': args.lr, 
            'weight_decay': args.weight_decay,
            'missing': args.missing,
            'seed': 'random',
            'numTest': args.numTest,
            'reconstruct': args.reconstruct
        }

    for test in range(args.numTest):
        if args.seed == 'random':
            setSeed = random.randint(1, 100001)
            info['seed'] = setSeed
        else:
            setSeed = int(args.seed)
        seed_everything(seed=setSeed)
        if args.log:
            sourceFile = open('./log.txt', 'a')
            print('*'*10, 'INFO' ,'*'*10, file = sourceFile)
            print(info, file = sourceFile)
            sourceFile.close()
        graphPath = './graphIEMOCAP.dgl'
        if args.missing > 0:
            graphPath = f'./graphIEMOCAP_missing_{args.missing}_test{test}.dgl'

        print("generating MM graph")
        if os.path.isfile(graphPath):
            (g,), _ = dgl.load_graphs(graphPath)
        else:
            if args.dataset == 'MELD':
                print("loading IEMOCAP")
            data = IEMOCAP(args.missing, args.edgeType)
            g = data[0]
            dgl.save_graphs(graphPath, g)
        print("loaded MM graph")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')
        g = g.to(device)
        features = g.ndata["feat"]
        labels = g.ndata["label"]
        masks = g.ndata["train_mask"], g.ndata["test_mask"]

        # create GCN model
        in_size = features.shape[1]
        out_size = 6
        if args.reconstruct:
            model = GAT_FP(in_size, 128, out_size).to(device)    
        else:
            model = GAT(in_size, 128, out_size).to(device)
        print(model)
        # model training
        print("Training...")
        highestAcc = train(g, features, labels, masks, model, info)

        # test the model
        print("Testing...")
        print(features.shape)
        acc = evaluate(g, features, labels, masks[1], model)
        print("Final Test accuracy {:.4f}".format(acc))

        if args.log:
            sourceFile = open('./log.txt', 'a')
            print(f'Highest Acc: {highestAcc}, final Acc {acc}', file = sourceFile)
            print('*'*10, 'End' ,'*'*10, file = sourceFile)
            sourceFile.close()