import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np 
import dgl.function as fn
from torch.autograd import Variable


class GATLayerVan(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayerVan, self).__init__()
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = nodes.mailbox['e']
        mean = torch.mean(alpha, 1)
        for ii in range(len(alpha)):
            alpha[ii][torch.where(alpha[ii] < mean[ii][0])[0]] = Variable(torch.tensor(0), requires_grad=False).cuda().float()
        #     tmp = torch.sum(alpha[ii])
        #     alpha[ii] = alpha[ii].clone() / tmp
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

class MultiHeadGATLayerVan(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayerVan, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayerVan(in_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# feat = torch.rand(6, 10)
# gatconv = MultiHeadGATLayerVan(10, 2, num_heads=3)
# res = gatconv(g, feat)
# print(res.shape)
# print(res)