import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb


def gnn_mask(h_prop_nodes, h_mask):
    inf_mask = h_mask.clone()
    inf_mask[h_mask == 0] = float('inf')
    inf_mask[h_mask == 1] = 0
    # Propagated [m, V, d]
    h_prop_masked_nodes = h_prop_nodes - inf_mask
    return h_prop_masked_nodes

class GgnnUndirectedEmbed(nn.Module):
    """
    Uses propagator for neural message passing phase

    """
    def __init__(self, num_steps, h_dim, V_size, N_edge_types, aggr_type=None):
        super(GgnnUndirectedEmbed, self).__init__()
        print("GNN number of prop steps=%d"%num_steps)
        if aggr_type is None:
            print("Use vanilla summation message passing")
            self._prop = GgnnUndirectedPropagator(h_dim, V_size, N_edge_types)
        else:
            print("Use non-standard summation aggregation for message passing")
            self._prop = AggrGgnnUndirectedPropagator(h_dim, V_size, N_edge_types, aggr_type)

        self._fcs = [nn.Linear(h_dim, h_dim) for _ in range(N_edge_types)]
        for i, layer in enumerate(self._fcs):
            self.add_module("%d"%i,layer)
        self._num_steps = num_steps
        self._N_edge_types = N_edge_types
        self._V_size = V_size
        self._h_dim = h_dim
        self._fc_init()
        self._aggr_type = aggr_type

    def _fc_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
        """
        for fc in self._fcs:
            fc.weight.data.normal_(0.0, 0.5)
        """

    def forward(self, h, A, h_query):
        """
        h: [m, V_size, d]
        A: [m, V_size, N_edge_types*V_size]
        returns: [m, V_size, d]
        """
        m = len(h)
        for i in range(self._num_steps):
            # h_adj is list(by N_edge_type) of [m, V_size, d] 
            h_adj_by_edge = []
            for j in range(self._N_edge_types):
                # [m, V_size, d] * [d, d2]
                h_adj_by_edge.append(self._fcs[j](h))
            # [m, N_edge, V_size, d] <- [N_edge, m, V_size, d]
            h_adj_by_edge = torch.stack(h_adj_by_edge).transpose(0, 1).contiguous() 
            # [m, N_edge*V_size, d]
            h_adj = h_adj_by_edge.view(m, self._N_edge_types*self._V_size, self._h_dim)
            # [m, V_size, d]
            if self._aggr_type == "query_attn":
                h = self._prop(h_adj, h, A, h_query)
            else:
                h = self._prop(h_adj, h, A)
        return h

    def get_attn_weights(self, h, A, h_query):
        m = len(h)
        attn_weights = []
        for i in range(self._num_steps):
            # h_adj is list(by N_edge_type) of [m, V_size, d] 
            h_adj_by_edge = []
            for j in range(self._N_edge_types):
                # [m, V_size, d] * [d, d2]
                h_adj_by_edge.append(self._fcs[j](h))
            # [m, N_edge, V_size, d] <= [N_edge, m, V_size, d]
            h_adj_by_edge = torch.stack(h_adj_by_edge).transpose(0, 1).contiguous() 
            # [m, N_edge*V_size, d]
            h_adj = h_adj_by_edge.view(m, self._N_edge_types*self._V_size, self._h_dim)
            attn_weights.append(self._prop.get_attn_weights(h_adj, A))
            # [m, V_size, d]
            if self._aggr_type == "query_attn":
                h = self._prop(h_adj, h, A, h_query)
            else:
                h = self._prop(h_adj, h, A)
        # [m, V, V]
        return attn_weights

    @property
    def embedding_dim(self):
        return self._h_dim

    @property
    def V_size(self):
        return self._V_size

    @property
    def N_edge_types(self):
        return self._N_edge_types


class GgnnUndirectedPropagator(nn.Module):
    def __init__(self, h_dim, V_size, N_edge_types):
        super(GgnnUndirectedPropagator, self).__init__()
        self._V_size = V_size
        self._N_edge_types = N_edge_types
        self._r_gate = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.Sigmoid()
                )
        self._u_gate = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.Sigmoid()
                )
        self._n_gate = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.Tanh()
                )
        self._h_dim = h_dim

    def forward(self, h_adj, h_prev, A):
        """
        h_adj, h_prev [m, V, d]
        A             [m, V_size, N_edge_types*V_size]
        (N_edge_types 1st dim, V_size 2nd dim for each N_edge_type]
        """
        # [m, V, d] 
        # = [m, V, V*N_edge_types]*[m, V*N_edge_types, d]
        h_adj_sums = torch.bmm(A, h_adj)
        # [m, V, 2d]
        a = torch.cat((h_adj_sums, h_prev), 2)

        # [m, V, d]
        r = self._r_gate(a)
        # [m, V, d]
        u = self._u_gate(a)
        h_hat = self._n_gate(torch.cat((h_adj_sums, r * h_prev), 2))
        # [m, V, d]
        h = (1 - u) * h_prev + u * h_hat
        return h


class AggrGgnnUndirectedPropagator(nn.Module):
    def __init__(self, h_dim, V_size, N_edge_types, aggr_type):
        super(AggrGgnnUndirectedPropagator, self).__init__()
        self._V_size = V_size
        self._N_edge_types = N_edge_types
        self._r_gate = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.Sigmoid()
                )
        self._u_gate = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.Sigmoid()
                )
        self._n_gate = nn.Sequential(
                nn.Linear(h_dim*2, h_dim),
                nn.Tanh()
                )
        self._h_dim = h_dim
        if aggr_type == "normalization":
            def atten_f(h_adj, A):
                attn_scores = A / (0.00001+A.sum(dim=2, keepdim=True))
                zero_vec = -9e15*torch.ones_like(attn_scores)
                attn_scores = torch.where(A>0, attn_scores, zero_vec)
                return attn_scores
        elif aggr_type == "attn":
            def atten_f(h_adj, A):
                attn_scores = torch.bmm(h_adj, torch.transpose(h_adj, 1, 2))
                zero_vec = -9e15*torch.ones_like(attn_scores)
                attn_scores = torch.where(A>0, attn_scores, zero_vec)
                attn_weights = F.softmax(attn_scores, dim=2)
                return attn_weights
        elif aggr_type == "relu_mlp_attn":
            self._fc = nn.Linear(h_dim, h_dim)
            def atten_f(h_adj, A):
                h_adj_W = self._fc(h_adj)
                # [m, V, V] = [m, V, d] x [m, d, V]
                attn_scores = F.leaky_relu(torch.bmm(h_adj_W, torch.transpose(h_adj, 1, 2)))
                zero_vec = -9e15*torch.ones_like(attn_scores)
                # attn weights mask by A
                attn_scores = torch.where(A>0, attn_scores, zero_vec)
                # [m, V, V(softmaxed)]
                attn_weights = F.softmax(attn_scores, dim=2)
                return attn_weights
        elif aggr_type == "self_attn":
            def atten_f(h_adj, A):
                pass
        elif aggr_type == "query_attn":
            def atten_f(h_adj, A, h_query):
                # h_query [m, d] h_adj [m, V, d], 
                # [m, 1, V] <= [m, 1, d]x[m, d, V]
                # ipdb.set_trace()
                attn_scores = torch.bmm(h_query.unsqueeze(1), torch.transpose(h_adj, 1, 2))
                # [m, V(copied), V] <= [m, 1, V]
                attn_scores = attn_scores.expand(-1, self._V_size, -1)
                zero_vec = -9e15*torch.ones_like(attn_scores)
                attn_scores = torch.where(A>0, attn_scores, zero_vec)
                attn_weights = F.softmax(attn_scores, dim=2)
                return attn_weights
        else:
            raise ValueError("Non_exist")
        self._aggr_type = aggr_type
        self._get_attn_weights_f = atten_f

    def get_attn_weights(self, h_adj, A):
        # [m, V, V]
        return self._get_attn_weights_f(h_adj, A)

    def forward(self, h_adj, h_prev, A, h_query=None):
        """
        h_adj, h_prev [m, V, d]
        A             [m, V_size, N_edge_types*V_size]
        (N_edge_types 1st dim, V_size 2nd dim for each N_edge_type]
        """
        # TYPEATTN
        # STEP 1 [m, V, V] <= [m, V, d] x [m, d, V]
        if h_query is not None:
            attn_weights = self._get_attn_weights_f(h_adj, A, h_query)
        else:
            attn_weights = self._get_attn_weights_f(h_adj, A)
        # [m, V, d] 
        # = [m, V, V*N_edge_types]*[m, V*N_edge_types, d]
        h_adj_sums = torch.bmm(attn_weights, h_adj)

        # [m, V, 2d]
        a = torch.cat((h_adj_sums, h_prev), 2)

        # [m, V, d]
        r = self._r_gate(a)
        # [m, V, d]
        u = self._u_gate(a)
        h_hat = self._n_gate(torch.cat((h_adj_sums, r * h_prev), 2))
        # [m, V, d]
        h = (1 - u) * h_prev + u * h_hat
        return h


def create_undirected_adj_matrix(doms, V_size):
    A = np.zeros((V_size, V_size), dtype=np.float32)
    adj_V_tokens = doms["adj_V"]
    for i, adj_V in enumerate(adj_V_tokens):
        for adj_idx in adj_V:
            A[i][adj_idx] = 1
    return A


# A [V, V] 
# Pos pairs: For each row, all the entries with 1
# Neg pairs: For each row, all the entries with 0
# Each row has a corresponding list of vals
def nce_pairs_gen(A, node_vals):
    assert len(A) == len(node_vals)
    pos_pairs = []
    V = len(A)
    for tgt_idx in range(V):
        for context_idx in range(V):
            if A[tgt_idx, context_idx] == 1:
                pos_pairs.append((node_vals[tgt_idx], node_vals[context_idx]))
    return pos_pairs














