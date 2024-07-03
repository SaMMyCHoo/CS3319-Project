import dgl.nn.pytorch as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl

import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        return x

class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.R = nn.ModuleList([LinearModel(input_dim, output_dim), LinearModel(input_dim, output_dim)])

    def forward(self, x):
        res = dict()
        for i, key in enumerate(x.keys()):
            feats = x[key]
            res[key] = self.R[i](feats)
        return res

def comb(x, y):
    res = dict()
    for key in x.keys():
        res[key] = x[key].view_as(y[key]) + y[key]
    return res

def select(block, f):
    res = dict()
    for key in f.keys():
        dst_nodes = block.dstnodes(key)
        res[key] = f[key][dst_nodes]
    return res

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.hid_feats = hid_feats
        self.out_feats = out_feats

        self.Res = nn.ModuleList([ResNet(in_feats, hid_feats)] + [ResNet(hid_feats, hid_feats) for _ in range(6)])
        
        self.Gat = nn.ModuleList([
            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats, self.hid_feats // 8, 8)
            for rel in rel_names}, aggregate='mean'),
            
            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 8, 8)
            for rel in rel_names}, aggregate='mean'), 
            
            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 4, 4)
            for rel in rel_names}, aggregate='mean'), 
            
            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 4, 4)
            for rel in rel_names}, aggregate='mean'), 

            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 4, 4)
            for rel in rel_names}, aggregate='mean'), 

            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 2, 2)
            for rel in rel_names}, aggregate='mean'), 

            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.hid_feats // 2, 2)
            for rel in rel_names}, aggregate='mean'), 
            
            dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(self.hid_feats, self.out_feats // 2, 2)
            for rel in rel_names}, aggregate='mean'), 
        ])

        self.bns = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for _ in range(7)])
        self.bns2 = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for _ in range(7)])

    def forward(self, blocks, inputs):
        r = self.Res[0](inputs)
        r = select(blocks[0], r)
        h = self.Gat[0](blocks[0], inputs)
        self.rel_list = list(h.keys())      
        h[self.rel_list[0]] = self.bns[0](h[self.rel_list[0]].view(-1, self.hid_feats))
        h[self.rel_list[1]] = self.bns2[0](h[self.rel_list[1]].view(-1, self.hid_feats))
        h = comb(h, r)
        h[self.rel_list[0]] = F.leaky_relu(h[self.rel_list[0]])
        h[self.rel_list[1]] = F.leaky_relu(h[self.rel_list[1]])

        for i in range(1, 7):
            r = self.Res[i](h)
            r = select(blocks[i], r)
            h = self.Gat[i](blocks[i], h)
            self.rel_list = list(h.keys())      
            h[self.rel_list[0]] = self.bns[i](h[self.rel_list[0]].view(-1, self.hid_feats))
            h[self.rel_list[1]] = self.bns2[i](h[self.rel_list[1]].view(-1, self.hid_feats))
            h = comb(h, r)
            if i <= 3:
                h[self.rel_list[0]] = F.leaky_relu(h[self.rel_list[0]])
                h[self.rel_list[1]] = F.leaky_relu(h[self.rel_list[1]])
            else:
                h[self.rel_list[0]] = F.tanh(h[self.rel_list[0]])
                h[self.rel_list[1]] = F.tanh(h[self.rel_list[1]])
                
        h = self.Gat[7](blocks[7], h)
        self.rel_list = list(h.keys())      
        h = {k: ((v.view(-1, self.out_feats))) for k, v in h.items()}
        
        return h

class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return edge_subgraph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, etypes):
        super().__init__()

        self.rgcn = RGCN(
            in_features, hidden_features, out_features, etypes)
        self.pred = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.rgcn(blocks, x)
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)
        return pos_score, neg_score