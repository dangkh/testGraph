import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from loadIEMOCAP import *
torch.set_default_dtype(torch.double)

# seed_everything(seed=10001)


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        gcv = [in_size, 512, 128, 32]
        self.layers = nn.ModuleList()
        # two-layer GCN
        for ii in range(len(gcv)-1):
            self.layers.append(
                dglnn.GraphConv(gcv[ii], gcv[ii+1], activation=F.relu)
            )
        # self.layers.append(dglnn.GraphConv(hid_size, 16))
        self.linear = nn.Linear(gcv[-1], out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = self.linear(h)
        return h

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

def auxilary(v):
    vv = v.clone().detach().cpu()
    listR = torch.zeros(6, 6)
    for ii in range(len(vv)):
        tmp = torch.unsqueeze(vv[ii], dim = 0)
        listR += tmp.T @ tmp
    mean = torch.sqrt(listR / len(vv))
    invMean = torch.linalg.inv(mean)
    vnew = v @ invMean.cuda()
    return vnew

def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        # logits = auxilary(logits)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 100.0 / len(labels)

def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # training loop
    for epoch in range(50):
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--2direction', action='store_true', default=False, help='edge direction type')
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    
    # g = dgl.add_self_loop(g)
    if os.path.isfile('./graphIEMOCAP.dgl'):
        (g,), _ = dgl.load_graphs('graphIEMOCAP.dgl')
    else:
        data = IEMOCAP()
        g = data[0]
        dgl.save_graphs('graphIEMOCAP.dgl', g)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = 6
    # model = GCN(in_size, 128, out_size).to(device)
    model = GAT(in_size, 128, out_size).to(device)
    print(model)
    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    print(features.shape)
    acc = evaluate(g, features, labels, masks[1], model)
    print("Test accuracy {:.4f}".format(acc))