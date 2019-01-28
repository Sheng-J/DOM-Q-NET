import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb


class NceNet(nn.Module):
    def __init__(self, V_max_size, E_dim):
        super(NceNet, self).__init__()
        self._E_dim = E_dim
        self._E_u = nn.Embedding(V_max_size, E_dim)
        self._E_v = nn.Embedding(V_max_size, E_dim)
        self._init_E()
        
    def _init_E(self):
        initrange = 0.5 / self._E_dim
        self._E_u.weight.data.uniform_(-initrange, initrange)
        self._E_v.weight.data.uniform_(-initrange, initrange)

    def forward(self, u, u_neg, v):
        m, k =  u_neg.size()
        # [m, d] <- [m]
        e_u = self._E_u(u)
        # [m, k, d] <- [m, k]
        e_u_neg = self._E_u(u_neg)
        # [m, d] <- [m]
        e_v = self._E_v(v)

        # [m] <- sum([m, d]*[m, d])
        score = torch.sum(e_u * e_v, dim=1)
        score = F.logsigmoid(score)

        # [m, k] = [m, k, n] * [m, n, 1]
        neg_score = torch.bmm(e_u_neg, e_v.unsqueeze(2)).squeeze(2)
        # (m*k scores) [m]
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)

        # [m] + [m]
        loss = (-1 * (score + neg_score)).mean()
        return loss

    def save_E(self, path):
        np.save(path, self._E_v.weight.detach().numpy())

    @property
    def E_state_dict(self):
        return self._E_v.state_dict()

    @property
    def E_weight(self):
        return self._E_v.weight.detach().cpu()

