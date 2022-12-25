import random 
import torch 
import numpy as np 
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
seed = 1001
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


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
    similar = 1.0 - torch.acos(cos(v1, v2))/ np.pi
    return similar

def convertNP2Tensor(listV):
    listR = []
    for xx in listV:
        listR.append(torch.from_numpy(xx))
    return listR

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.5, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

def norm(features):
    meanMat = np.mean(features, axis=0, keepdims=True)
    stdMat = np.std(features, axis=0, keepdims=True)
    stdMat[np.where(stdMat == 0)] = 1
    newFeatures = (features - meanMat) / stdMat
    # minMat = np.min(newFeatures, axis = 0, keepdims=True)
    # newFeatures = newFeatures- minMat
    return newFeatures

def auxilary(v):
    vv = v.clone().detach().cpu()
    meanMat = np.mean(np.asarray(vv), axis=0, keepdims=True)
    stdMat = np.std(np.asarray(vv), axis=0, keepdims=True)
    stdMat[np.where(stdMat == 0)] = 1
    newFeatures = (np.asarray(vv) - meanMat) / stdMat
    minMat = np.min(newFeatures, axis = 0, keepdims=True)
    newFeatures = torch.from_numpy(newFeatures- minMat)
    # newFeatures = torch.from_numpy(newFeatures)
    # listR = torch.zeros(vv[0].shape[0], vv[0].shape[0])
    # for ii in range(len(newFeatures)):
    #     tmp = torch.unsqueeze(newFeatures[ii], dim = 0)
    #     listR += tmp.T @ tmp
    mean = torch.sqrt(torch.abs(newFeatures.T @ newFeatures) / len(newFeatures.T))
    invMean = torch.linalg.inv(mean)
    vnew = v @ invMean.cuda()
    return vnew

def vis(info):
    print('Visualize')
    X0, y0 = info
    visData = [[X0, y0]]
    embeddings = visData[0][0]
    tsne = TSNE(n_components=2)
    transformed = tsne.fit_transform(embeddings)

    palette = sns.color_palette("bright", len(np.unique(y0)))
    g = sns.scatterplot(
        x=transformed[:,0],
        y=transformed[:,1],
        hue=visData[0][1],
        legend='full',
        palette=palette
    )
    # _lg = g.get_legend()
    # _lg.remove()
    plt.show()

def vis2(info):
    print('Visualize')
    X0, y0 = info
    visData = [[X0, y0]]
    embeddings = visData[0][0]
    tsne = TSNE(n_components=3)
    transformed = tsne.fit_transform(embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = ListedColormap(sns.color_palette().as_hex())
    sc = ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], s=10, c=visData[0][1], marker='o', cmap=cmap, alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def evaluate(g, mask, model):
    model.eval()
    features = g.ndata['feat']
    labels = g.ndata['label']
    with torch.no_grad():
        logits = model(g, features)
        mask = mask.bool()
        # logits = auxilary(logits)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 100.0 / len(labels)
        return f1_score(indices.cpu(), labels.cpu(), average='weighted')

def evaluateMSE(g, features, labels, mask, model, visual = False, originalLabel = None):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        d = logits - labels
        meanSquareD = torch.sqrt(torch.mean(d**2, dim=0)).sum().item()
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        if visual:
            logits = auxilary(logits)
            originalLabel = originalLabel[mask]
            vis2([logits.cpu(), originalLabel.cpu()])
        return meanSquareD

def convertX2Binary(labels):
    listLabel = []
    sizeLabel = len(np.unique(labels.cpu()))
    for ll in labels:
        newll = torch.zeros(sizeLabel)
        newll[ll] = 1
        listLabel.append(newll)
    r = torch.stack(listLabel )
    return r.cuda()
