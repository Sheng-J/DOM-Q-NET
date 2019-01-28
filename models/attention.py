import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import ipdb


class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass





#class ScaledDotProductAttention(nn.Module):
#    def __init__(self, d_k):
#        super(ScaledDotProductAttention, self).__init__()
#        self._d_k = d_k
#        self.sqrt_dk = np.power(d_k, 0.5)
#
#    def forward(self, Q, K, V, attn_mask=None):
#        """
#        Q [m, num_q_tokens, d_k]
#        K [m, num_kv_tokens, d_k]
#        V [m, num_kv_tokens, d_v]
#        -> [m, num_q_tokens, d_v]
#        """
#        # [m, num_q_tokens, num_kv_tokens] = 
#        # [m, num_q_tokens, d_k] * [m, d_k, num_kv_tokens]
#        attn = torch.bmm(Q, K.transpose(1, 2)) / self.sqrt_dk
#
#        m, num_q_tokens, num_kv_tokens = attn.size()
#
#        attn = attn.view(m*num_q_tokens, num_kv_tokens)
#        # [m*num_q_tokens, num_kv_tokens]
#        attn = F.softmax(attn, dim=1)
#        # [m, num_q_tokens, num_kv_tokens[attns]]
#        # "how important each kv is for a query"
#        attn = attn.view(m, num_q_tokens, num_kv_tokens)
#
#        # [m, num_q_tokens, d_V] = [m, num_q_tokens, num_kv_tokens]*[m, num_kv_tokens, d_V]
#        output = torch.bmm(attn, V)
#
#        return output, attn


def scaled_dot_attn(Q, K, V, attn_mask):
    """
    Q [m, num_q_tokens, d_k]  (m, 1, d_k)
    K [m, num_kv_tokens, d_k] (m, max_V, d_k)
    V [m, num_kv_tokens, d_v] (m, max_V, d_k)
    attn_mask [m, num_q_tokens, num_kv_tokens] (m, 1, max_V)
    (K == V here)

    -> [m, num_q_tokens, d_v]  [m, 1, d_k]
    """
    # [m, num_q_tokens, num_kv_tokens] = 
    # [m, num_q_tokens, d_k] * [m, d_k, num_kv_tokens]
    d_k = Q.size()[2]
    attn = torch.bmm(Q, K.transpose(1, 2)) / (d_k**0.5)
    m, num_q_tokens, num_kv_tokens = attn.size()

    # MASK out pads
    inf_mask = attn_mask.clone()
    inf_mask[attn_mask == 0] = float('inf')
    inf_mask[attn_mask == 1] = 0
    attn = attn - inf_mask

    attn = attn.view(m*num_q_tokens, num_kv_tokens)
    # [m*num_q_tokens, num_kv_tokens]
    attn = F.softmax(attn, dim=1)
    # [m, num_q_tokens, num_kv_tokens[attns]]
    # "how important each kv is for a query"
    attn = attn.view(m, num_q_tokens, num_kv_tokens)

    # [m, num_q_tokens, d_V] = [m, num_q_tokens, num_kv_tokens]*[m, num_kv_tokens, d_V]
    output = torch.bmm(attn, V)

    return output, attn

