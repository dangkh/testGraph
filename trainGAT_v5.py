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
from AttentionModuleVan import *
from AttentionInnerModule import *
# torch.set_default_dtype(torch.float)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class maskFilter(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        tt, aa, vv  = 100, 442, 2024
        if in_size == 1247:
            tt, aa, vv  = 600, 942, 1242
        # self.testM = nn.Parameter(torch.rand(in_size, in_size))
        currentFeatures = np.asarray([0.0] * in_size)
        textMask = np.copy(currentFeatures)
        textMask[:tt] = 1.0
        audioMask = np.copy(currentFeatures)
        audioMask[tt: aa] = 1.0
        videoMask = np.copy(currentFeatures)
        videoMask[aa:] = 1.0
        self.textMask = torch.from_numpy(textMask) * torch.tensor(3.0)
        self.textMask = nn.Parameter(self.textMask).float().to(device)
        
        self.audioMask = torch.from_numpy(audioMask) * torch.tensor(2.0)
        self.audioMask = nn.Parameter(self.audioMask).float().to(device)
        
        self.videoMask = torch.from_numpy(videoMask) * torch.tensor(1.0)
        self.videoMask = nn.Parameter(self.videoMask).float().to(device)


    def forward(self, features):
        return features * self.textMask + features * self.audioMask + features * self.videoMask

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.textMask.item()} + {self.audioMask.item()} + {self.videoMask.item()}'

class GAT_FP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, wFP, numFP):
        super().__init__()
        gcv = [in_size, 256, 8]
        self.maskFilter = maskFilter(in_size)
        self.num_heads = 4
        # self.GATFP = dglnn.GATv2Conv(in_size,  in_size,  num_heads = 4)
        self.GATFP = dglnn.GraphConv(in_size,  in_size, norm = 'both', weight=False, bias = False)
        self.gat1 = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.gat1.append(
                dglnn.GATv2Conv(np.power(self.num_heads, ii) * gcv[ii],  gcv[ii+1], activation=F.relu,  residual=True, num_heads = self.num_heads)
            )
        coef = 1
        self.gat2 = MultiHeadGATInnerLayer(in_size,  gcv[-1], num_heads = self.num_heads)
        # self.layers.append(dglnn.GraphConv(hid_size, 16))
        self.linear = nn.Linear(gcv[-1] * self.num_heads * 2, out_size)
        # self.linear = nn.Linear(gcv[-1] * self.num_heads * 7, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features.float()
        mask = torch.zeros(h.shape)
        missIndx = torch.where(features==0)
        mask[missIndx] = 1
        # h1 = self.GATFP(g, h)
        meanF = torch.mean(h, 0)
        meanF = meanF.repeat(len(h), 1)
        meanF = meanF.double()
        h1= features.clone()
        h1[missIndx] =  meanF[missIndx]
        h = 0.5 * (h + h1)
        h = h.float()
        # h = h + h1
        h = F.normalize(h, p=1)
        h = self.maskFilter(h)
        h3 = self.gat2(g, h)
        for i, layer in enumerate(self.gat1):
            if i != 0:
                h = self.dropout(h)
            h = h.float()
            h = torch.reshape(h, (len(h), -1))
            h = layer(g, h)
        
        h = torch.reshape(h, (len(h), -1))
        h = torch.cat((h3,h), 1)
        h = self.linear(h)
        return h


def label2negative(indx, lenL):
    r = random.randint(0, lenL)
    while r != indx.item():
        # print(r, indx.item())
        r = random.randint(0, lenL)
        return r
    return r


def train(g, trainSet, testSet, masks, model, info):
    # define train/val samples, loss function and optimizer
    if info['MSE'] == False:
        loss_fcn = nn.CrossEntropyLoss()
        # loss_fcn = FocalLoss()
    else:
        loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=info['lr'], weight_decay=info['weight_decay'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=info['lr'], momentum=1.0)
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
    parser.add_argument('--batchSize', help='size of batch', default=1, type=int)
    parser.add_argument('--mergeLabel', help='if True then mergeLabel from 6 to 4',action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=True, help='save experiment info in output')
    parser.add_argument('--output', help='savedFile', default='./log.txt')
    parser.add_argument('--prePath', help='prepath to directory contain DGL files', default='F:/dangkh/work/dgl')
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
            'seed': args.seed,
            'numTest': args.numTest,
            'wFP': args.wFP,
            'numFP': args.numFP,
            'MSE': args.MSE
        }
    for test in range(args.numTest):
        if args.seed == 'random':
            setSeed = seedList[test]
            info['seed'] = setSeed
        else:
            setSeed = int(args.seed)
        seed_everything(seed=setSeed)
        if args.log:
            sourceFile = open(args.output, 'a')
            print('*'*10, 'INFO' ,'*'*10, file = sourceFile)
            print(info, file = sourceFile)
            sourceFile.close()
        graphPath = f'{args.prePath}/graph{args.dataset}/{args.dataset}.dgl'
        trainPath = f'{args.prePath}/graph{args.dataset}/{args.dataset}_trainset.dgl'
        testPath = f'{args.prePath}/graph{args.dataset}/{args.dataset}_testset.dgl'
        if args.missing > 0:
            graphPath = f'{args.prePath}/graph{args.dataset}/missing_{args.missing}_test{test}.dgl'
            trainPath = f'{args.prePath}/graph{args.dataset}/missing_{args.missing}_test{test}_trainset.dgl'
            testPath = f'{args.prePath}/graph{args.dataset}/missing_{args.missing}_test{test}_testset.dgl'
            
        print("generating MM graph")
        if os.path.isfile(graphPath):
            (g,), _ = dgl.load_graphs(graphPath)
            (trainSet), _ = dgl.load_graphs(trainPath)
            (testSet,), _ = dgl.load_graphs(testPath)
            trainSet = dgl.batch(trainSet)
        else:        
            dataPath  = './IEMOCAP_features/IEMOCAP_features.pkl'
            if args.dataset == 'MELD':
                dataPath  = './MELD_features/MELD_features.pkl'
            data = IEMOCAP(missing = args.missing, nameDataset = args.dataset, path = dataPath)
            g, trainSet, testSet = data[0]
            dgl.save_graphs(graphPath, g)
            dgl.save_graphs(trainPath, dgl.unbatch(trainSet))
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
        model = GAT_FP(in_size, 128, out_size, args.wFP, args.numFP).to(device)    
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # stop
        print(model)
        # model training
        print("Training...")
        highestAcc = train(g, trainSet, testSet, masks, model, info)
        # test the model
        print("Testing...")
        print(features.shape)
        if args.MSE:
            acc = evaluateMSE(g, features, labels, masks[1], model, visual=True, originalLabel=g.ndata["label"])
        else:
            labels = g.ndata["label"]
            acc = evaluate(g, masks[1], model)
            # labels = convertX2Binary(labels)
            # evaluateMSE(g, features, labels, masks[0], model, visual=True, originalLabel=g.ndata["label"])
        print("Final Test accuracy {:.4f}".format(acc))

        if args.log:
            sourceFile = open(args.output, 'a')
            print(f'Highest Acc: {highestAcc}, final Acc {acc}', file = sourceFile)
            print('*'*10, 'End' ,'*'*10, file = sourceFile)
            sourceFile.close()