import torch
import torch.nn as nn
import numpy as np
import ipdb


class Embedding(nn.Module):
    def __init__(self, V, E_dim):
        super(Embedding, self).__init__()
        self._E_dim = E_dim
        self._E = nn.Embedding(V.max_size, E_dim)
        self._V = V

    def get_status(self):
        status = {
                "E": self._E.weight,
                "V": str(self._V)
                }
        return status

    def prep(self, x, max_num_1d_tokens, oov2randidx_dict=None):
        """
        token_ids [max_num_1d_tokens]
        num_1d_tokens: [1]
        mask [max_num_1d_tokens]
        """
        # num_1d_tokens = [len(x)]
        num_1d_tokens = min(len(x), max_num_1d_tokens)
        # [1, len(unpadded_tokens)]
        tokens = [x]
        if oov2randidx_dict is None:
            token_ids, mask_1d = batch_pad_lookup_(
                    tokens, self._V, max_num_1d_tokens
                    )
            return token_ids.squeeze(0), num_1d_tokens, mask_1d.squeeze(0)
        else:
            # [1, max_num_1d_tokens], [1, s], [1, s], [1, s]
            token_ids, mask_1d, oov_mask, oov_ids = batch_pad_oov_lookup_(
                    tokens, self._V, oov2randidx_dict, max_num_1d_tokens
                    )
            # [max_num_1d_tokens], 1, [max_num_1d_tokens], [max_num_1d_tokens]
            return token_ids.squeeze(0), num_1d_tokens, mask_1d.squeeze(0), oov_mask.squeeze(0), oov_ids.squeeze(0)

    def forward(self, X):
        """
        [m, max_num_1d_tokens]
        -> [m, max_num_1d_tokens, d]
        """ 
        return self._E(X)

    def rev_prep(self, x):
        return self._V.rev_lookups(x.cpu().numpy())

    @property
    def embedding_dim(self):
        return self._E_dim

    @property
    def track_info(self):
        return {"V_size": len(self._V)}

    @property
    def embed_input_dim(self):
        return 2

    def fork_E_2D(self, pad_elem_1d):
        return Embedding2D(self._V, self._E, pad_elem_1d)

    @property
    def weight(self):
        return self._E.weight

    def get_E_cos_dist_mat(self, num_rows):
        E = self._E.weight.detach()[:num_rows]
        nume = torch.matmul(E, E.t())
        norm_E = E.norm(dim=1, keepdim=True)
        deno = torch.matmul(norm_E, norm_E.t())
        deno = torch.max(deno, torch.ones_like(deno)*1e-8)
        cos_dist_mat = (nume / deno).cpu().numpy()
        return cos_dist_mat


class Embedding2D(nn.Module):
    def __init__(self, V, E, pad_elem_1d):
        """
        should be forked from a shared Embedding1D
        """
        super(Embedding2D, self).__init__()
        self._E_dim = E.embedding_dim
        self._E = E
        self._V = V  # V could be a vocab for 'text' attr chars
        self._pad_elem_1d = pad_elem_1d

    def get_status(self):
        status = {
                "E": self._E.weight,
                "V": str(self._Vchar)
                }
        return status

    def prep(self, X, max_num_1d_tokens, max_num_2d_tokens):
        """
        X strings [unpadded(num_1d_tokens), unpadded(string)]
        returns 
        - token_ids [max_num_doms, padded_max_chars_len] (required by forward)
        - seq_lens [max_num_doms]
        - mask [max_num_doms, padded_max_chars_len]
        (mask is mostly useless due to seq_lens provided)
        BEWARE MASK BY THE CALLER
        """
        X_ = []
        for x in X:
            # seq_lens.append(len(x))
            X_.append([letter for letter in x])
            if len(X_) == max_num_1d_tokens:
                break
        num_1d_tokens = [len(X_)]
        while len(X_) < max_num_1d_tokens:
            X_.append(self._pad_elem_1d)
        # [max_num_doms, unpadded_len(chars)]

        # [max_num_doms, max_chars_len]
        token_ids, mask = batch_pad_lookup_(X_, self._V, max_num_2d_tokens)
        # [max_num_doms]
        # TODO need to debug why so many 10s
        num_2d_tokens = [int(sum(char_masks)) for char_masks in mask]
        # [max_num_doms, max_chars_len], [max_num_doms],
        # [max_num_doms, max_chars_len]
        return token_ids, num_1d_tokens, num_2d_tokens

    def forward(self, X):
        """
        [m, max_num_doms, max_chars_len]
        -> [m, max_num_doms, max_chars_len, d]
        OR
        [m*max_num_doms, max_chars_len]
        -> [m*max_num_doms, max_chars_len, d]
        """
        return self._E(X)

    @property
    def embedding_dim(self):
        return self._E_dim

    @property
    def track_info(self):
        return {"V_size": len(self._V)}

    @property
    def embed_input_dim(self):
        return 3

    @property
    def weight(self):
        return self._E.weight


def batch_pad_(batch_tokens, pad_token, max_num_tokens=None):
    """
    In-place
    Output: mask [m, s]
    """
    if max_num_tokens is None:
        max_num_tokens = max(len(sub_tokens) for sub_tokens in batch_tokens)
    mask = np.ones((len(batch_tokens), max_num_tokens), dtype=np.float32)
    for sub_tokens, submask in zip(batch_tokens, mask):
        if len(sub_tokens) < max_num_tokens:
            submask[len(sub_tokens):max_num_tokens] = 0.
            sub_tokens.extend([pad_token]*(max_num_tokens - len(sub_tokens)))
        elif len(sub_tokens) > max_num_tokens:
            while len(sub_tokens)!=max_num_tokens:
                sub_tokens.pop()
    return mask


def batch_pad_lookup_(batch_tokens, vocab, max_num_tokens=None):
    """
    Inputs: batch_tokens: list of list
    Outputs: batch_ids [m, s] np.int64
             mask [m, s] np.float32
         are np.ndarray objects
    """
    # [m, s]
    mask = batch_pad_(batch_tokens, vocab.pad_token, max_num_tokens)
    ids = []
    for sub_tokens in batch_tokens:
        ids.append([])
        for x in sub_tokens:
            x_id = vocab[x]
            ids[-1].append(x_id)
    # TODO CHECK np type
    batch_ids = np.array(ids)
    return batch_ids, mask


def batch_pad_oov_lookup_(batch_tokens, vocab, oov2randidx_dict=None, max_num_tokens=None):
    """
    Inputs: batch_tokens: list of list
    Outputs: batch_ids [m, s] np.int64
             mask [m, s] np.float32
         are np.ndarray objects
    """
    # [m, s]
    mask = batch_pad_(batch_tokens, vocab.pad_token, max_num_tokens)
    ids = []
    oov_mask = []
    # oov_idx & oov2randidx_dict make sure oov tokens in different place can
    # have same "rand" embeddings (E.g. goal, dom)
    oov_idxs = []
    for sub_tokens in batch_tokens:
        ids.append([])
        oov_mask.append([])
        oov_idxs.append([])
        for x in sub_tokens:
            x_id = vocab[x]
            ids[-1].append(x_id)
            if x_id != vocab.unk_id:
                # not out of vocab
                oov_mask[-1].append(0)
            else:
                # OOV 
                oov_mask[-1].append(1)
                if x not in oov2randidx_dict:
                    num_oov_items = len(oov2randidx_dict)
                    oov2randidx_dict[x] = num_oov_items
                oov_idxs[-1].append(oov2randidx_dict[x])
    # TODO CHECK np type
    batch_ids = np.array(ids, dtype=np.int64)
    # [m, max_num_tokens]
    oov_mask = np.array(oov_mask)
    oov_ids = np.array(oov_idxs, dtype=np.int64)
    return batch_ids, mask, oov_mask, oov_ids



