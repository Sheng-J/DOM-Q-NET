import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from plp.utils.np_wrap import batch_pad_lookup_
import ipdb


class EmbedNet(nn.Module):
    def __init__(self, h_dim, max_num_1d_tokens, E, h0_batch_device=None):
        super(EmbedNet, self).__init__()
        self._E = E
        # self._lstm = nn.LSTM(E.embedding_dim, h_dim)
        self._h_dim = h_dim
        self._max_num_1d_tokens = max_num_1d_tokens
        self._h0_device = h0_batch_device

    @property
    def max_seq_len(self):
        return self._max_num_1d_tokens

    def get_status(self):
        status = {
            "E": self._E.weight
            }
        return status

    def prep(self, x, oov2randidx_dict=None):
        """
        x : a list of sequential items
        E.g ["press", "okay", "button"] or ['S', 'e', 'l', 'e']
        -> [max_num_1d_tokens], [1], [max_num_1d_tokens]
        """
        return self._E.prep(x, self._max_num_1d_tokens, oov2randidx_dict)
        #x_ids, num_1d_tokens, mask = self._E.prep(x, self._max_num_1d_tokens)
        #return x_ids, num_1d_tokens, mask

    def forward(self, X_ids, X_num_1d_tokens):
        """
        [m, max_num_tokens] -> [m, d]
        """
        # Sort x
        m = len(X_ids)
        max_num_1d_tokens = len(X_ids[0])
        # X_ids [m, seq_len]
        old_idx__seq_len = sorted(
                enumerate(X_num_1d_tokens), key=lambda xi: xi[1], reverse=True
                )
        new2old_idxs = []
        for new_idx, (old_idx, num_tokens) in enumerate(old_idx__seq_len):
            new2old_idxs.append(old_idx)
        X_ = X_ids[new2old_idxs]
        lengths = X_num_1d_tokens[new2old_idxs]
        # lengths = list(X_num_1d_tokens[new2old_idxs].cpu().numpy())
        X = self._E(X_)
        x_packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)

        if self._h0_device is not None:
            h0 = (
                    torch.zeros(1, m, self._h_dim, device=self._h0_device, requires_grad=False),
                    torch.zeros(1, m, self._h_dim, device=self._h0_device, requires_grad=False)
                 )
        else:
            h0 = (
                    torch.zeros(1, m, self._h_dim, requires_grad=False),
                    torch.zeros(1, m, self._h_dim, requires_grad=False)
                 )

        y_packed, h = self._lstm(x_packed, h0)
        y, lens = nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
        idxs = (lens -1).view(m, 1).expand(m, self._h_dim).unsqueeze(1)
        if self._h0_device is not None:
            idxs = idxs.to(self._h0_device)
        y_last = y.gather(1, idxs).squeeze(1)

        # Unsort
        old2new_idxs = [new2old_idxs.index(i) for i in range(m)]
        y_last = y_last[old2new_idxs]
        return y_last

    @property
    def embedding_dim(self):
        return self._h_dim

    @property
    def embed_input_dim(self):
        return 2


class EmbedNet2D(nn.Module):
    """
    Flatten[m*V_size, seq_len] -> Embed [m*V_size, seq_len, d]
    -> RNN [m*V_size, seq_len, d] -> Last hidden [m*V_size, d]
    -> Unflatten [m, V_size, d]
    """
    def __init__(
            self, h_dim, max_num_2d_tokens, E_2D, h0_device=None
            ):
        super(EmbedNet2D, self).__init__()
        self._E_2D = E_2D
        self._lstm = nn.LSTM(E_2D.embedding_dim, h_dim)
        self._h_dim = h_dim
        self._max_num_2d_tokens = max_num_2d_tokens
        self._h0_device = h0_device

    def get_status(self):
        status = {
            "E": self._E_2D.weight
            }
        return status

    def prep(self, x, max_num_1d_tokens):
        """
        X: a list of strings/sequential items (unpadded to max_num_doms)
        E.g. [num_1d_tokens(unpadded), num_2d_tokens(unpadded_string)]
        -> [max_num_1d_tokens, max_num_2d_tokens]
        """
        x_ids, num_1d_tokens, num_2d_tokens = self._E_2D.prep(
                x, max_num_1d_tokens, self._max_num_2d_tokens
                )
        return x_ids, num_1d_tokens, num_2d_tokens

    def forward(self, X_ids, X_num_2d_tokens):
        """
        [m, V_size, max_num_chars] -> [m, V_size, d]
        """
        # X_seq_lens [batch_size, max_num_doms] -> [batch_size * max_num_doms]
        # Flatten list [batch_size, max_num_doms, max_num_text_chars]
        # into X_ids [batch_size * max_num_doms, max_num_text_chars]
        m = len(X_ids)
        max_num_1d_tokens = len(X_ids[0])
        big_m = m*max_num_1d_tokens
        # assert len(X_ids[0]) == len(X_seq_lens[0])
        X_ids = X_ids.view(big_m, self._max_num_2d_tokens)
        X_num_2d_tokens = X_num_2d_tokens.view(big_m)
        # 
        # X batch of list of 1 tensor,1 val ([seq_id0, seq_id1, .]], seq_len])
        # Sort x  xi[1][1], 1st dimension due to enumerate idx
        old_idx__seq_lens = sorted(
                enumerate(X_num_2d_tokens),
                key=lambda xi: xi[1], reverse=True
                )
        new2old_idxs = []
        for new_idx, (old_idx, num_2d_tokens) in enumerate(old_idx__seq_lens):
            new2old_idxs.append(old_idx)
        X_ = X_ids[new2old_idxs]
        # [m*max_num_doms]
        lengths = X_num_2d_tokens[new2old_idxs]
        # [m*max_num_doms, max_chars_len] -> [m*max_num_doms, max_chars_len, d]
        X = self._E_2D(X_)
        x_packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)

        if self._h0_device is not None:
            h0 = (
                    torch.zeros(1, big_m, self._h_dim, device=self._h0_device, requires_grad=False),
                    torch.zeros(1, big_m, self._h_dim, device=self._h0_device, requires_grad=False)
                 )
        else:
            h0 = (
                    torch.zeros(1, big_m, self._h_dim, requires_grad=False),
                    torch.zeros(1, big_m, self._h_dim, requires_grad=False)
                 )

        y_packed, h = self._lstm(x_packed, h0)
        # y[m*max_num_doms, seq_len, d]
        y, lens = nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
        # last h idxs [m*max_num_doms] -> [m*max_num_doms, 1] -> [m*max_num_doms,(copied)d]
        # -> [m*max_num_doms, 1, d]
        idxs = (lens - 1).view(big_m, 1).expand(big_m, self._h_dim).unsqueeze(1)
        if self._h0_device is not None:
            idxs = idxs.to(self._h0_device)
        # y_last[m*max_num_doms, d]
        y_last = y.gather(1, idxs).squeeze(1)
        # unsort_idxes
        old2new_idxs = [new2old_idxs.index(i) for i in range(big_m)]
        y_last = y_last[old2new_idxs]
        # [m, max_num_doms, d]
        y_last = y_last.view(m, max_num_1d_tokens, self._h_dim)
        return y_last

    @property
    def embedding_dim(self):
        return self._h_dim

    @property
    def embed_input_dim(self):
        return 3
    @property
    def max_seq_len(self):
        return self._max_num_2d_tokens

