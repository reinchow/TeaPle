import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_

class SE_GNN(nn.Module):
    def __init__(self, h_dim, n_ent, n_rel, kg_n_layer, pred_rel_w=False, out_channel=32, ker_sz=3, ent_drop=0.1, rel_drop=0.1):
        super(SE_GNN, self).__init__()
        self.h_dim = h_dim
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.kg_n_layer = kg_n_layer


        self.ent_emb = nn.Embedding(n_ent, h_dim)

        # GNN layers
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim) for _ in range(kg_n_layer)])
        self.node_layers = nn.ModuleList([NodeLayer(h_dim) for _ in range(kg_n_layer)])
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(kg_n_layer)])


        self.rel_embs = nn.ParameterList([nn.Parameter(torch.randn(n_rel * 2, h_dim)) for _ in range(kg_n_layer)])

        if pred_rel_w:
            self.rel_w = nn.Parameter(torch.randn(h_dim * kg_n_layer, h_dim))
        else:
            self.pred_rel_emb = nn.Parameter(torch.randn(n_rel * 2, h_dim))


        self.ent_drop = nn.Dropout(ent_drop)
        self.rel_drop = nn.Dropout(rel_drop)
        self.act = nn.Tanh()

    def forward(self, h_id, r_id, kg):

        ent_emb, rel_emb = self.aggragate_emb(kg)

        head = ent_emb[h_id]
        rel = rel_emb[r_id]

        return head, rel, ent_emb

    def aggragate_emb(self, kg):
        ent_emb = self.ent_emb.weight
        rel_emb_list = []
        for edge_layer, node_layer, comp_layer, rel_emb in zip(self.edge_layers, self.node_layers, self.comp_layers, self.rel_embs):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, rel_emb)
            ent_emb = ent_emb + edge_ent_emb + node_ent_emb + comp_ent_emb
            rel_emb_list.append(rel_emb)

        if hasattr(self, 'rel_w'):
            pred_rel_emb = torch.cat(rel_emb_list, dim=1)
            pred_rel_emb = pred_rel_emb.mm(self.rel_w)
        else:
            pred_rel_emb = self.pred_rel_emb

        return ent_emb, pred_rel_emb


class FTuckER(nn.Module):
    def __init__(self, entity_dim, relation_dim, rank, input_dropout=0.1, hidden_dropout1=0.1, hidden_dropout2=0.1):
        super(FTuckER, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.rank = rank


        self.core_tensor = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank, rank, rank)), 
                                        dtype=torch.float, requires_grad=True))

        self.entity_emb = nn.Embedding(entity_dim, rank)
        self.relation_emb = nn.Embedding(relation_dim, rank)

        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout1 = nn.Dropout(hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(hidden_dropout2)

        self.bn0 = nn.BatchNorm1d(entity_dim)
        self.bn1 = nn.BatchNorm1d(entity_dim)

        xavier_normal_(self.entity_emb.weight.data)
        xavier_normal_(self.relation_emb.weight.data)

    def forward(self, head, relation, tail):

        head = self.bn0(head)
        head = self.input_dropout(head)

        head = torch.matmul(head, self.entity_emb.weight)
        relation = torch.matmul(relation, self.relation_emb.weight)

        core_output = torch.einsum('abc,br,cr,ar->b', self.core_tensor, head, relation, tail)
        core_output = F.leaky_relu(core_output, negative_slope=0.2)  # 引入 LeakyReLU
        core_output = self.hidden_dropout1(core_output)
        core_output = self.bn1(core_output)
        core_output = self.hidden_dropout2(core_output)

        pred = torch.sigmoid(core_output)
        return pred


class GNN_FTuckER(nn.Module):
    def __init__(self, h_dim, n_ent, n_rel, kg_n_layer, entity_dim, relation_dim, rank, pred_rel_w=False):
        super(GNN_FTuckER, self).__init__()
        self.encoder = SE_GNN(h_dim, n_ent, n_rel, kg_n_layer, pred_rel_w)
        self.decoder = FTuckER(entity_dim, relation_dim, rank)

    def forward(self, h_id, r_id, kg, tail_id):

        head, rel, ent_emb = self.encoder(h_id, r_id, kg)


        tail = ent_emb[tail_id]
        pred = self.decoder(head, rel, tail)

        return pred