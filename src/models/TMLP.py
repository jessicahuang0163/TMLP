# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender
from utils.graphmlp import GMLP


class TMLP(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TMLP, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.knn_k = config["knn_k"]
        self.n_ui_layers = config["n_ui_layers"]
        self.reg_weight = config["reg_weight"]
        self.mm_image_weight = config["mm_image_weight"]
        self.dropout = config["dropout"]
        self.hidden_dim = config["hidden_dim"]
        self.v_dropout = config["v_dropout"]
        self.t_dropout = config["t_dropout"]
        self.num_fc_layers = config["num_fc_layers"]
        self.act_fn = config["act_fn"]
        self.tau1 = config["tau1"]
        self.alpha1 = config["alpha1"]

        self.n_nodes = self.n_users + self.n_items

        has_id = True
        num_user = self.n_users
        num_item = self.n_items
        batch_size = config["train_batch_size"]
        dim_x = config["embedding_size"]
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = config["aggr_mode"]
        self.dataset = dataset
        self.num_layer = 1
        self.drop_rate = 0.1

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.mm_adj = None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(
            self.device
        ), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        mm_adj_file = os.path.join(
            dataset_path,
            "mm_adj_freedomdsp_{}_{}.pt".format(
                self.knn_k, int(10 * self.mm_image_weight)
            ),
        )

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=False
            )
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=False
            )

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(
                    self.image_embedding.weight.detach()
                )
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(
                    self.text_embedding.weight.detach()
                )
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = (
                    self.mm_image_weight * image_adj
                    + (1.0 - self.mm_image_weight) * text_adj
                )
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form="coo").astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        )
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]

        self.gcn = GCN(
            self.dataset,
            batch_size,
            num_user,
            num_item,
            dim_x,
            self.aggr_mode,
            num_layer=self.num_layer,
            has_id=has_id,
            dropout=self.drop_rate,
            dim_latent=64,
            device=self.device,
            features=self.v_feat,
        )
        self.MLP_v = nn.Linear(self.v_feat.size(1), 2 * self.feat_embed_dim)
        self.MLP_v1 = nn.Linear(2 * self.feat_embed_dim, self.feat_embed_dim)
        self.MLP_t = nn.Linear(self.t_feat.size(1), 2 * self.feat_embed_dim)
        self.MLP_t1 = nn.Linear(2 * self.feat_embed_dim, self.feat_embed_dim)

        self.vgmlp = GMLP(
            self.embedding_dim,
            self.hidden_dim,
            self.v_dropout,
            output_dim=self.embedding_dim,
            num_fc_layers=self.num_fc_layers,
            act_fn=self.act_fn,
        )
        self.tgmlp = GMLP(
            self.embedding_dim,
            self.hidden_dim,
            self.t_dropout,
            output_dim=self.embedding_dim,
            num_fc_layers=self.num_fc_layers,
            act_fn=self.act_fn,
        )
        self.v_preference, self.t_preference = None, None
        self.adj_tensor = torch.load(config["adj_tensor_path"]).to(self.device)

    def pre_epoch_processing(self):
        pass

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(
            torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True)
        )
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self):
        v_feat = self.MLP_v1(F.leaky_relu(self.MLP_v(self.v_feat)))
        t_feat = self.MLP_t1(F.leaky_relu(self.MLP_t(self.t_feat)))
        id_feat = self.item_id_embedding.weight
        v_feat = self.vgmlp(v_feat)
        t_feat = self.tgmlp(t_feat)
        all_feat = torch.cat((v_feat, t_feat, id_feat), dim=1)
        all_rep, self.preference = self.gcn(self.edge_index, self.edge_index, all_feat)
        item_rep = all_rep[self.num_user :]
        user_rep = all_rep[: self.num_user]

        return user_rep, item_rep

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward()

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )
        item_idx = torch.cat((pos_items, neg_items))
        ii_dis = self.get_feature_dis(ia_embeddings[item_idx])
        ii_adj = self.adj_tensor.to_dense()[item_idx, :][:, item_idx]
        ncloss1 = self.alpha1 * self.Ncontrast(ii_dis, ii_adj, tau=self.tau1)
        reg_embedding_loss_v = (
            (self.v_preference[users] ** 2).mean()
            if self.v_preference is not None
            else 0.0
        )
        reg_embedding_loss_t = (
            (self.t_preference[users] ** 2).mean()
            if self.t_preference is not None
            else 0.0
        )
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)

        return batch_mf_loss + reg_loss + ncloss1

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def Ncontrast(self, x_dis, adj_label, tau=1):
        """
        compute the Ncontrast loss
        """
        x_dis = torch.exp(tau * x_dis)
        x_dis_sum = torch.sum(x_dis, 1)
        x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
        loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
        return loss

    def get_feature_dis(self, x):
        """
        x :           batch_size x nhid
        x_dis(i,j):   item means the similarity between x(i) and x(j).
        """
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / x_norm
        x_dis = x @ x.T
        mask = torch.eye(x_dis.shape[0], device=x.device)
        x_dis = (1 - mask) * x_dis
        return x_dis


class GCN(torch.nn.Module):
    def __init__(
        self,
        datasets,
        batch_size,
        num_user,
        num_item,
        dim_id,
        aggr_mode,
        num_layer,
        has_id,
        dropout,
        dim_latent=None,
        device=None,
        features=None,
    ):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        self.preference = nn.Parameter(
            nn.init.xavier_normal_(
                torch.tensor(
                    np.random.randn(num_user, self.dim_latent * 3),
                    dtype=torch.float32,
                    requires_grad=True,
                ),
                gain=1,
            ).to(self.device)
        )
        self.conv_embed_1 = Base_gcn(
            self.dim_latent, self.dim_latent, aggr=self.aggr_mode
        )

    def forward(self, edge_index_drop, edge_index, features):
        x = torch.cat((self.preference, features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat = h + x + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(
        self, in_channels, out_channels, normalize=True, bias=True, aggr="add", **kwargs
    ):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == "add":
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return "{}({},{})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
