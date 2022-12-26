import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np 
import dgl.function as fn

class GATInnerLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATInnerLayer, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.qMask = nn.Linear(in_dim, in_dim, bias=False)
        self.kMask = nn.Linear(in_dim, in_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.qMask.weight, gain=gain)
        nn.init.xavier_normal_(self.kMask.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        qVal = self.qMask(edges.src['h'])
        qVal = torch.unsqueeze(qVal, 2)
        kVal = self.kMask(edges.src['h'])
        kVal = torch.unsqueeze(kVal, 1)
        score = (qVal @ kVal)  / np.sqrt(self.in_dim)
        score = F.softmax(score, dim=1)
        origin = torch.unsqueeze(edges.src['h'], 2)
        attVal = torch.bmm(score, origin).sum(dim = 2)
        # print(attVal.shape)
        return {'a': attVal, 's': score}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'a': edges.data['a'], 's': edges.data['s']}

    def reduce_func(self, nodes):
        # self.counter += 1
        # print(f"debug counter at: {self.counter}")
        # print(nodes.mailbox)
        # datas =nodes.mailbox['a'] 

        # print(datas.shape)
        # print(score.shape)
        # lenFeature = datas.shape[-1]
        # datas = datas.reshape(-1, lenFeature)
        # print(datas.shape)
        out = torch.mean(nodes.mailbox['a'] , dim = 1)
        # print(out.shape)
        return {'h': out}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_attention)

        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATInnerLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATInnerLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATInnerLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
g = dgl.add_self_loop(g)
feat = torch.rand(6, 10)
gatconv = MultiHeadGATInnerLayer(g, 10, 2, num_heads=3)
res = gatconv(feat)
# print(res.shape)
# print(res)