import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph import create_undirected_adj_matrix, gnn_mask
from models.layers import NoisyLinear
from models.attention import scaled_dot_attn
from pprint import pprint
import ipdb


class Qnet(nn.Module):
    def __init__(
            self, E_dom, E_ggnn, max_num_leaves, h_dom_dim, num_atoms=None,
            use_c51=False, dueling_type=None, use_noisylayers=False,
            use_goal_attn=True, use_goal_cat=False,
            use_local=True, use_neighbor=True, use_global=True
            ):
        """
        Dueling - Type1:  embed1,2,3->V, A
                - Type2:  embed1,2->A  3->V
        NoisyLinear - Replace fc2 and out with noisy layers
        Note: a_dim is not given, max_num_leaves max num of actions is given
        E_x: - DomLeavesEmbedding
        E_gnn:
        """
        super(Qnet, self).__init__()
        self._use_c51 = use_c51
        self._dueling_type = dueling_type
        self._use_noisylayers = use_noisylayers
        self._num_atoms = num_atoms
        self._max_num_leaves = max_num_leaves  
        self._E_dom = E_dom
        self._E_dom_dim = E_dom.E_dom_dim
        self._E_ggnn = E_ggnn 
        self._max_num_goal_tokens = self._E_dom.max_num_goal_tokens
        self._use_goal_attn = use_goal_attn
        self._use_goal_cat = use_goal_cat
        self._use_local = use_local
        self._use_neighbor = use_neighbor
        self._use_global = use_global

        if self._use_c51:
            print("Using C51 NET")
            assert num_atoms is not None
        if self._use_noisylayers:
            print("Use Noisy")

        if dueling_type is not None:
            stream_dim = int(fc_dim/2)
            # stream_dim = fc_dim
            print("Use Dueling type %d with each stream dim=%d"%(dueling_type, stream_dim))
            if dueling_type == 0:
                self._body_net = DuelingBody0(
                        max_num_leaves, self._E_dom_dim, stream_dim, use_noisylayers
                        )
            elif dueling_type == 1:
                self._body_net = DuelingBody1(
                        max_num_leaves, self._E_dom_dim, stream_dim, use_noisylayers
                        )
            if use_c51:
                self._head_net = DuelingC51Head(
                        max_num_leaves, stream_dim, use_noisylayers, num_atoms
                        )
            else:
                self._head_net = DuelingHead(
                        max_num_leaves, stream_dim, use_noisylayers
                        )
        else:
            if use_goal_attn:
                if use_goal_cat:
                    print("use goal_attn_cat")
                    in_dom_global_dim = self._E_dom_dim*3 + self._E_dom_dim
                else:
                    print("use goal_attn")
                    in_dom_global_dim = self._E_dom_dim*3 
            else:
                if use_goal_cat:
                    print("use goal_cat")
                    in_dom_global_dim = self._E_dom_dim*2 + self._E_dom_dim
                else:
                    print("no attn or cat")
                    in_dom_global_dim = self._E_dom_dim*2

            token_in_dim = in_dom_global_dim + E_dom.text_dim + self._max_num_goal_tokens
            self._body_net = Body(
                    in_dom_global_dim,
                    max_num_leaves, self._max_num_goal_tokens,
                    self._E_dom_dim, token_in_dim,
                    h_dom_dim, use_noisylayers, use_local, use_neighbor, use_global 
                    )
            if use_c51:
                self._head_net = C51Head(
                        max_num_leaves, fc_dim, use_noisylayers, num_atoms
                        )
            else:
                self._head_net = Head(
                        max_num_leaves, self._max_num_goal_tokens, h_dom_dim, use_noisylayers
                        )

    def prep(self, x):
        # x from environment
        dom_vals, goal, _ = x
        # The followings are padded to V
        # 2d: top_tokens, tag_ids, text_ids, classes_ids, focus_encodes,
        # tampered_encodes, V_mask, goal_ids, token_positions, goal_mask = self._E_x.prep(dom_vals, goal)
        # 1d: goal_oov_mask, goal_oov_ids, text_oov_mask, text_oov_ids
        prep_list2d, prep_list1d = self._E_dom.prep(dom_vals, goal)
        prep_list2d = list(prep_list2d)

        V_size = len(prep_list2d[0])
        A = create_undirected_adj_matrix(dom_vals, V_size) # [V, V]

        is_leaves = dom_vals["is_leaf"]
        leaves_idxs = [i for i in range(len(is_leaves)) if is_leaves[i]]
        leaves_mask = [1.0 for _ in range(len(leaves_idxs))]

        max_num_leaves, num_leaves = self._max_num_leaves, len(leaves_mask)
        assert max_num_leaves >= num_leaves  

        pad_masked_idx = len(is_leaves) - 1
        leaves_idxs = leaves_idxs + [pad_masked_idx for _ in range(max_num_leaves - num_leaves)] 
        leaves_idxs = np.array(leaves_idxs)
        leaves_idxs_mask = leaves_mask + [0.0 for _ in range(max_num_leaves - num_leaves)]
        leaves_idxs_mask = np.array(leaves_idxs_mask, dtype=np.float32)
        #                         + [V, V], [max_num_leaves], [max_num_leaves]
        prep_list2d = prep_list2d + [A, leaves_idxs, leaves_idxs_mask]
        return prep_list2d, prep_list1d

    def forward(self, X, log=False):
        # 1. top_tokens, tag_ids, text_ids, classes_ids, focus_encodes,
        # tampered_encodes, 
        # 2. A, leaves_idxs, leaves_mask
        # 3. V_mask, goal_ids, goal_seq_lens, A, leaves_idxs, leaves_mask
        embed_prep_list, (A, leaves_idxs, leaves_mask), embed_prep_list1d = X[:-7], X[-7:-4], X[-4:]
        m = len(embed_prep_list[0])
        leaves_E_idxs = leaves_idxs.unsqueeze(2).expand(-1, -1, self._E_dom_dim)

        # [m, d_h_goal], _, [m, max_num_goal_tokens], [m, V, E_dim], [m, V_size]<= 10+4 args
        h_goal, embedded_goal_tokens, goal_mask, e_local, V_mask = self._E_dom(*embed_prep_list, *embed_prep_list1d)
        # [m, V_size, E_dom_dim] <= [m, V]
        V_E_mask = V_mask.unsqueeze(2).expand(-1, -1, self._E_dom_dim)
        V_size = len(e_local)
        # Message Passing: [m, V_size, E_dom] <= [m, V, E_dom]
        e_neighbor = self._E_ggnn(e_local, A, h_goal)

        # [m, max_num_leaves, E_dom] <= [m, V, E_dom] 
        e_local_leaves = e_local.gather(dim=1, index=leaves_E_idxs)
        e_neighbor_leaves = e_neighbor.gather(dim=1, index=leaves_E_idxs)

        # [m, E_dom] <= [m, max_num_leaves, E_dom]
        e_global_max_from_local = e_local.max(dim=1)[0]
        e_global_max_from_neighbor = e_neighbor.max(dim=1)[0]

        if self._use_goal_attn:
            # [m, max_num_leaves, 2d]
            e_global_attn, attns = scaled_dot_attn(
            h_goal.unsqueeze(1), e_local, e_neighbor, V_mask.unsqueeze(1)
                )
            # [m, self._E_dom_dim], [m, V_size]
            e_global_attn, attns = e_global_attn.squeeze(1), attns.squeeze(1)
            if self._use_goal_cat:
                e_global = torch.cat((
                    e_global_max_from_local, e_global_max_from_neighbor, 
                    e_global_attn,
                    h_goal),dim=1)
            else:
                e_global = torch.cat((
                    e_global_max_from_local, e_global_max_from_neighbor,
                    e_global_attn,
                    ), dim=1)
        else:
            if self._use_goal_cat:
                e_global = torch.cat((e_global_max_from_local, e_global_max_from_neighbor, h_goal),dim=1)
            else:
                e_global = torch.cat((e_global_max_from_local, e_global_max_from_neighbor), dim=1)

        # Goal G state
        # [m, max_num_goal(copied), d_h_goal(max_num_goal_tokens)]
        expanded_e_global = e_global.unsqueeze(1).expand(
                -1, self._max_num_goal_tokens, -1)
        e_tokens = torch.cat((embedded_goal_tokens, expanded_e_global), dim=2)

        # Final representation for Q network
        e_list = self._body_net(e_local_leaves, e_neighbor_leaves, e_global, e_tokens)
        return self._head_net(e_list, leaves_mask, goal_mask, log)

    def get_attn_weights(self, X):
        tag_ids, text_ids, classes_ids, focus_encodes, V_mask, goal_ids, goal_seq_lens, A, leaves_idxs, leaves_mask = X
        # [m, V_size, x_dom_dim], []
        X_dom, _ = self._E_dom(tag_ids, text_ids, classes_ids, focus_encodes, V_mask, goal_ids, goal_seq_lens)
        # [m, V_size, V_size]
        tag_tokens = self._E_dom.rev_prep(tag_ids[0])
        return self._E_ggnn.get_attn_weights(X_dom, A), A, tag_tokens

    def debug_h(self, x):
        return {}, {}
        raise NotImplementedError("needs update implem")
        return {}, {"h":self._embed([x])[0].squeeze(0).cpu().detach().numpy()}

    @property
    def use_noisylayers(self):
        return self._use_noisylayers

    def reset_noise(self):
        assert self._use_noisylayers
        self._body_net.reset_noise()
        self._head_net.reset_noise()


##
# Body Modules
##
#
#class DuelingBody0(nn.Module):
#    def __init__(self, max_num_leaves, in_dim, stream_dim, use_noisylayers):
#        super(DuelingBody0, self).__init__()
#        self._h_V_dim = in_dim * 4
#        # self._h_V_dim = in_dim
#        self._h_A_dim = in_dim * 2
#        self._max_num_leaves = max_num_leaves
#        if use_noisylayers:
#            self._V_fc1 = NoisyLinear(self._h_V_dim, stream_dim)
#            self._A_fc1 = NoisyLinear(self._h_A_dim, stream_dim)
#            self._V_fc2 = NoisyLinear(stream_dim, stream_dim)
#            self._A_fc2 = NoisyLinear(stream_dim, stream_dim)
#        else:
#            self._V_fc1 = nn.Linear(self._h_V_dim, stream_dim)
#            self._A_fc1 = nn.Linear(self._h_A_dim, stream_dim)
#            self._V_fc2 = nn.Linear(stream_dim, stream_dim)
#            self._A_fc2 = nn.Linear(stream_dim, stream_dim)
#
#    def reset_noise(self):
#        self._V_fc1.reset_noise()
#        self._V_fc2.reset_noise()
#        self._A_fc1.reset_noise()
#        self._A_fc2.reset_noise()
#
#    def forward(self, h_leaves, h_prop_leaves, h_V):
#        h_V = F.relu(self._V_fc1(h_V))
#        # [m, stream_dim]
#        h_V = F.relu(self._V_fc2(h_V))
#
#        # [m, max_num_leaves 2*d]
#        h_A = torch.cat((h_leaves, h_prop_leaves), dim=2)
#        h_A = h_A.view(-1, self._h_A_dim)
#        h_A = F.relu(self._A_fc1(h_A))
#        # [m*max_num_leaves, stream_dim]
#        h_A = F.relu(self._A_fc2(h_A))
#        return h_V, h_A
#
#
#class DuelingBody1(nn.Module):
#    def __init__(self, max_num_leaves, in_dim, stream_dim, use_noisylayers):
#        super(DuelingBody1, self).__init__()
#        #self._h_V_dim = in_dim
#        #self._h_A_dim = in_dim * 3
#        self._h_V_dim = in_dim * 4
#        self._h_A_dim = in_dim * 6
#        self._max_num_leaves = max_num_leaves
#        if use_noisylayers:
#            self._V_fc1 = NoisyLinear(self._h_V_dim, stream_dim)
#            self._A_fc1 = NoisyLinear(self._h_A_dim, stream_dim)
#            self._V_fc2 = NoisyLinear(stream_dim, stream_dim)
#            self._A_fc2 = NoisyLinear(stream_dim, stream_dim)
#        else:
#            self._V_fc1 = nn.Linear(self._h_V_dim, stream_dim)
#            self._A_fc1 = nn.Linear(self._h_A_dim, stream_dim)
#            self._V_fc2 = nn.Linear(stream_dim, stream_dim)
#            self._A_fc2 = nn.Linear(stream_dim, stream_dim)
#
#    def reset_noise(self):
#        self._V_fc1.reset_noise()
#        self._V_fc2.reset_noise()
#        self._A_fc1.reset_noise()
#        self._A_fc2.reset_noise()
#
#    def forward(self, h_leaves, h_prop_leaves, h_V):
#        h_max = h_V.unsqueeze(1).expand(-1, self._max_num_leaves, -1)
#        h_A = torch.cat((h_leaves, h_prop_leaves, h_max), dim=2)
#        h_V = F.relu(self._V_fc1(h_V))
#        # [m, stream_dim]
#        h_V = F.relu(self._V_fc2(h_V))
#
#        # [m, max_num_leaves 3*d]
#        h_A = h_A.view(-1, self._h_A_dim) 
#        h_A = F.relu(self._A_fc1(h_A))
#        # [m * max_num_leaves, stream_dim]
#        h_A = F.relu(self._A_fc2(h_A))
#        return h_V, h_A


class Body(nn.Module):
    def __init__(self,
                 in_dom_global_dim,
                 max_num_leaves,
                 max_num_goal_tokens,
                 in_dom_dim,
                 in_token_dim,
                 h_dom_dim,
                 use_noisylayers,
                 use_local,
                 use_neighbor,
                 use_global):
        super(Body, self).__init__()
        assert use_local or use_neighbor
        self._use_local = use_local
        self._use_neighbor = use_neighbor
        self._use_global = use_global
        if not use_global:
            if use_local and use_neighbor:
                self._in_cat_dom_dim = in_dom_dim * 2
            else:
                self._in_cat_dom_dim = in_dom_dim
        else:
            if use_local and use_neighbor:
                self._in_cat_dom_dim = in_dom_dim * 2 + in_dom_global_dim
            else:
                self._in_cat_dom_dim = in_dom_dim + in_dom_global_dim

        self._in_mode_dim = in_dom_global_dim
        self._in_token_dim = in_token_dim
        self._max_num_leaves = max_num_leaves
        self._max_num_goal_tokens = max_num_goal_tokens
        h_mode_dim, h_token_dim = int(h_dom_dim/4), int(h_dom_dim/2)
        if use_noisylayers:
            self._fc1_dom = NoisyLinear(self._in_cat_dom_dim, h_dom_dim)
            self._fc2_dom = NoisyLinear(h_dom_dim, h_dom_dim)
            self._fc1_mode = NoisyLinear(self._in_mode_dim, h_mode_dim)
            self._fc2_mode = NoisyLinear(h_mode_dim, h_mode_dim)
            self._fc1_token = NoisyLinear(self._in_token_dim, h_token_dim)
            self._fc2_token = NoisyLinear(h_token_dim, h_token_dim)
        else:
            pass

    def forward(self, e_local, e_neighbor, e_global, e_tokens):
        # [m, d]
        e_global_ = e_global
        # [m, max_num_leaves(copied), d]<-[m, d]
        e_global = e_global.unsqueeze(1).expand(-1, self._max_num_leaves, -1)
        # [m * self._max_num_leaves, 3*d]
        if not self._use_global:
            if self._use_local and self._use_neighbor:
                e_dom = torch.cat((e_local, e_neighbor), dim=2)
            elif not self._use_local:
                e_dom = e_neighbor
            else:
                e_dom = e_local
        else:
            if self._use_local and self._use_neighbor:
                e_dom = torch.cat((e_local, e_neighbor, e_global), dim=2)
            elif not self._use_local:
                e_dom = torch.cat((e_neighbor, e_glboal), dim=2)
            else:
                e_dom = torch.cat((e_local, e_global), dim=2)
        # [1] DOM stream
        # [m*V_size, in_dom_dim] <- [m, V_size, in_dom_dim]
        e_dom = e_dom.view(-1, self._in_cat_dom_dim)
        h_dom = F.relu(self._fc1_dom(e_dom))
        h_dom = F.relu(self._fc2_dom(h_dom))
        # [2] Mode stream
        h_mode = F.relu(self._fc1_mode(e_global_))
        h_mode = F.relu(self._fc2_mode(h_mode))
        # [3] Token Stream
        # [m*max_num_tokens, in_token_dim] <- [m, max_num_tokens, in_token_dim]
        e_tokens = e_tokens.view(-1, self._in_token_dim)
        h_tokens = F.relu(self._fc1_token(e_tokens))
        h_tokens = F.relu(self._fc2_token(h_tokens))
        return h_dom, h_mode, h_tokens

    def reset_noise(self):
        self._fc1_dom.reset_noise()
        self._fc2_dom.reset_noise()
        self._fc1_mode.reset_noise()
        self._fc2_mode.reset_noise()
        self._fc1_token.reset_noise()
        self._fc2_token.reset_noise()


## 
# Output Head Modules
##

class Head(nn.Module):
    def __init__(
            self,
            max_num_leaves,
            max_num_goal_tokens,
            dom_stream_dim,
            use_noisylayers):
        super(Head, self).__init__()
        self._max_num_leaves = max_num_leaves
        self._max_num_goal_tokens = max_num_goal_tokens
        mode_stream_dim, token_stream_dim = int(dom_stream_dim/4), int(dom_stream_dim/2)
        if use_noisylayers:
            self._out_dom = NoisyLinear(dom_stream_dim, 1)
            self._out_mode = NoisyLinear(mode_stream_dim, 2)
            self._out_token = NoisyLinear(token_stream_dim, 1)
        else:
            self._out_dom = nn.Linear(dom_stream_dim, 1)
            self._out_mode = nn.Linear(mode_stream_dim, 2)
            self._out_token = nn.Linear(token_stream_dim, 1)

    def forward(self, h_list, leaves_mask, goal_mask, _):
        """
        Outputs: Q_dom for dom, Q_type for click or type(binary),
                 Q_token for selected goal token idx
        """
        h_dom, h_mode, h_token = h_list
        Q_dom = self._out_dom(h_dom)
        # 1. [m, max_num_leaves] <= [m*max_num_leaves]
        Q_dom = Q_dom.view(-1, self._max_num_leaves)
        inf_mask = leaves_mask.clone()
        inf_mask[leaves_mask == 0] = float('inf')
        inf_mask[leaves_mask == 1] = 0
        Q_dom -= inf_mask
        # 2. [m, 2]
        Q_mode = self._out_mode(h_mode)

        Q_token = self._out_token(h_token)
        # 3. [m, max_num_tokens] <= [m*max_num_tokens]
        Q_token = Q_token.view(-1, self._max_num_goal_tokens)
        inf_mask = goal_mask.clone()
        inf_mask[goal_mask == 0] = float('inf')
        inf_mask[goal_mask == 1] = 0
        # [m, max_num_tokens]
        Q_token -= inf_mask
        return Q_dom, Q_mode, Q_token

    def reset_noise(self):
        self._out_dom.reset_noise()
        self._out_mode.reset_noise()
        self._out_token.reset_noise()


class C51Head(nn.Module):
    def __init__(self, max_num_leaves, in_dim, use_noisylayers, num_atoms):
        super(C51Head, self).__init__()
        self._max_num_leaves = max_num_leaves
        self._num_atoms = num_atoms
        if use_noisylayers:
            self._out = NoisyLinear(in_dim, self._num_atoms)
        else:
            self._out = nn.Linear(in_dim, self._num_atoms)

    def forward(self, h, leaves_mask, log=False):
        Y = self._out(h).view(-1, self._max_num_leaves, self._num_atoms)
        if log:
            P = F.log_softmax(Y, dim=2)
        else:
            P = F.softmax(Y, dim=2)
        # [m, max_num_leaves, num_atoms], [m, max_num_leaves]
        return P, leaves_mask

    def reset_noise(self):
        self._out.reset_noise()


class DuelingHead(nn.Module):
    def __init__(self, max_num_leaves, in_dim, use_noisylayers):
        super(DuelingHead, self).__init__()
        self._max_num_leaves = max_num_leaves
        if use_noisylayers:
            self._V_out = NoisyLinear(in_dim, 1)
            self._A_out = NoisyLinear(in_dim, 1)
        else:
            self._V_out = nn.Linear(in_dim, 1)
            self._A_out = nn.Linear(in_dim, 1)

    def forward(self, h, leaves_mask, _):
        h_V, h_A = h
        # [m, 1]
        V = self._V_out(h_V)
        # [m, max_num_leaves]
        A = self._A_out(h_A).view(-1, self._max_num_leaves)
        # [m, max_num_leaves]
        Q = V + A - ((A*leaves_mask).sum(1, keepdim=True) / leaves_mask.sum(1, keepdim=True))
        inf_mask = leaves_mask.clone()
        inf_mask[leaves_mask == 0] = float('inf')
        inf_mask[leaves_mask == 1] = 0
        Q -= inf_mask
        return Q

    def reset_noise(self):
        self._V_out.reset_noise()
        self._A_out.reset_noise()


class DuelingC51Head(nn.Module):
    def __init__(self, max_num_leaves, in_dim, use_noisylayers, num_atoms):
        super(DuelingC51Head, self).__init__()
        self._max_num_leaves = max_num_leaves
        self._num_atoms
        if use_noisylayers:
            self._V_out = NoisyLinear(in_dim, 1*self._num_atoms)
            self._A_out = NoisyLinear(in_dim, 1*self._num_atoms)
        else:
            self._V_out = nn.Linear(in_dim, self._num_atoms)
            self._A_out = nn.Linear(in_dim, self._num_atoms)

    def forward(self, h, leaves_mask, log=False):
        h_V, h_A = h
        # [m, 1, num_atoms]
        V = self._V_out(h_V).view(-1, 1, self._num_atoms)
        # [m, max_num_leaves, num_atoms]
        leaves_mask_expanded = leaves_mask.unsqueeze(2).expand(-1, -1, self._num_atoms)
        # [m, max_num_leaves, num_atoms]
        A = self._A_out(h_A).view(-1, self._max_num_leaves, self._num_atoms)
        # [m, max_num_leaves, num_atoms]
        Y = V + A - ((A*leaves_mask_expanded).sum(1, keepdim=True) / leaves_mask_expanded.sum(1, keepdim=True))
        if log:
            P = F.log_softmax(Y, dim=2)
        else:
            P = F.softmax(Y, dim=2)
        return P, leaves_mask

    def reset_noise(self):
        self._V_out.reset_noise()
        self._A_out.reset_noise()









