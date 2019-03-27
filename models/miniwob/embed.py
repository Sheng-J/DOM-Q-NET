import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from algorithms.encodings import position_encoding_init
from models.attention import scaled_dot_attn
import ipdb


class DomLeavesEmbedding(nn.Module):
    """
    Embed top or not
    E_dom_dim:  cos_align + E_tag + E_classes + E_focus + E_tampered + E_top
    [1] dot product for E_text_embed and E_q_embed

    bgColor: rgb(85, 85, 85), fgColor: rgb(0, 0, 0)
    left:0, top: 0, width: 640
    """
    def __init__(
            self, 
            E_tag,
            E_text,
            E_classes,
            E_q,
            max_num_doms,
            batch_device,
            embed_text=False,
            embed_pos=False,
            embed_color=False):
        super(DomLeavesEmbedding, self).__init__()
        self._E_tag = E_tag
        self._E_text = E_text
        self._E_q = E_q
        self._E_classes = E_classes
        self._max_num_doms = max_num_doms
        E_align_dim = 1
        # E_align_dim = 2 # NOTE cos_dist + pos_enc_cos_dist
        E_focus_embeedding_dim = 2
        E_tampered_embeedding_dim = 2
        E_top_embedding_dim = 1
        self._embed_text = embed_text
        self._embed_pos = embed_pos
        self._embed_color = embed_color
        extra_dim = 0
        if embed_pos:
            print("Embedding pos attr")
            extra_dim += 3  # top, left, width
            self._embed_pos = embed_pos
        if embed_color:
            print("Embedding color attr")
            extra_dim += 6  # 3bg rgb, 3fg rgb
            self._embed_color = embed_color
        self._E_dom_dim =  E_align_dim + E_tag.embedding_dim \
            + E_classes.embedding_dim + E_focus_embeedding_dim \
            + E_tampered_embeedding_dim + extra_dim
        self._E_text_dim = E_text.embedding_dim

        # self._lin_proj = nn.Linear(E_text.embedding_dim, self._E_align_dim)
        self._oov_rand_device = batch_device
        # POS ENCODING
        # self._max_num_goal_tokens = max_num_goal_tokens
        n_pos, d_text = E_q.max_seq_len, self._E_text_dim
        self._max_num_goal_tokens = n_pos

        self._pos_E = torch.eye(n_pos, n_pos, device=batch_device).unsqueeze(0)
        # max_num_goal_tokens for pos encoding
        self._goal_fc = nn.Linear(n_pos+d_text, self._E_dom_dim)

    @property
    def max_num_goal_tokens(self):
        return self._max_num_goal_tokens

    @property
    def is_graph_embedding(self):
        return False

    def get_status(self):
        status = {
                "E_tag": self._E_tag.get_status(),
                "E_text": self._E_text.get_status(),
                "E_classes": self._E_classes.get_status()
            }
        return status

    def prep(self, x, goal):
        # [num_doms]
        tag_tokens, text_tokens, classes_tokens, focus_tokens, tampered_tokens \
                = x["tag"], x["text"], x["classes"], x["focused"], x["tampered"]
        top_tokens = x["top"]
        # top_tokens, left_tokens, width_tokens = x["top"], x["left"], x["width"]
        #fg_color, bg_color = x["fg_color"][4:-1].split(","), x["bg_color"][4:-1].split(",")
        # Get actual num_doms before padded
        num_doms = len(tag_tokens)
        assert self._max_num_doms - num_doms >= 0

        # Handle oov for text only for now.
        tag_ids, _, V_mask = self._E_tag.prep(tag_tokens, self._max_num_doms)
        classes_ids, _, _ = self._E_classes.prep(classes_tokens, self._max_num_doms)
        focus_encodes = focus_tokens[:self._max_num_doms] + [[1.0, 0.0] for _ in range(self._max_num_doms-num_doms)]
        tampered_encodes = tampered_tokens[:self._max_num_doms] + [[1.0, 0.0] for _ in range(self._max_num_doms-num_doms)]
        top_tokens = top_tokens[:self._max_num_doms] + [[0.0] for _ in range(self._max_num_doms-num_doms)]

        text_oov2randidx_dict = {}
        assert self._max_num_goal_tokens - len(goal)>=0
        goal_ids, _max_num_goal_tokens, goal_mask, goal_oov_mask, goal_oov_ids = self._E_q.prep(goal, text_oov2randidx_dict)  # [N]

        token_positions = np.array([pos_i+1 for pos_i in range(self._E_q.max_seq_len)])
        # [max_num_doms] for word embeddings
        text_ids, __num_doms, __doms_mask, text_oov_mask, text_oov_ids = self._E_text.prep(text_tokens, self._max_num_doms, text_oov2randidx_dict)
        return (top_tokens, tag_ids, text_ids, classes_ids, focus_encodes, \
          tampered_encodes, V_mask, goal_ids, token_positions, goal_mask), \
          (goal_oov_mask, goal_oov_ids, text_oov_mask, text_oov_ids)

    def forward(
            self,
            top_tokens,
            tag_ids,
            text_ids,
            classes_ids,
            focus_encodes,
            tampered_encodes,
            V_mask,
            goal_ids,
            token_positions,
            goal_mask,
            goal_oov_mask,
            goal_oov_ids,
            text_oov_mask,
            text_oov_ids):
        m = len(goal_ids)
        # [m, V/max_num_doms, d_tag]
        tag_embeds = self._E_tag(tag_ids)
        # [m, V/max_num_doms, d_text]
        text_embeds = self._E_text(text_ids)
        # [m, V/max_num_doms, d_class]
        classes_embeds = self._E_classes(classes_ids)
        max_num_goal_tokens = len(goal_ids[0])
        # [m, max_num_goal_tokens, d_text]
        goal_embeds = self._E_text(goal_ids)
        # HANDLE OOV
        text_embeds, goal_embeds = self._oov_rand_mask(text_embeds, text_oov_mask, text_oov_ids, goal_embeds, goal_oov_mask, goal_oov_ids)

        # goal aggregated representation 
        pos_enc = self._pos_E.expand(m, -1, -1)
        # [m, max_num_goal_tokens, d_text+max_num_goal_tokens]
        pos_goal_embeds = torch.cat((goal_embeds, pos_enc), dim=2)
        # [m, max_num_goal_tokens, d_h_goal] <= [m, max_num_goal_tokens]
        goal_embed_mask = goal_mask.unsqueeze(2).expand(-1, -1, self._E_dom_dim)
        # [m, max_num_goal_tokens, d_h_goal]
        masked_goal_vec = goal_embed_mask * F.relu(self._goal_fc(pos_goal_embeds))
        # [m, d_h_goal]
        h_goal = (masked_goal_vec).max(dim=1)[0]

        # Module: COSINE SIMILARITY ALIGNMENT
        assert self._max_num_goal_tokens == len(goal_ids[0])
        # [m, V/max_num_doms(COPIED), max_num_goal_tokens, d_text]
        expanded_goal_embeds = goal_embeds.unsqueeze(1).expand(-1, self._max_num_doms, -1, -1)
        # [m, V/max_num_doms, max_num_goal_tokens(COPIED), d_text]
        expanded_text_embeds = text_embeds.unsqueeze(2).expand(-1, -1, self._max_num_goal_tokens, -1)
        # DOM-Goal cos align [m, V/max_num_doms, max_num_goal_tokens]
        cos_dist = F.cosine_similarity(expanded_goal_embeds, expanded_text_embeds, dim=3)
        # [m, V/max_num_doms, 1] <= [m, V/max_num_doms, max_num_goal_tokens]
        alignments = (torch.max(cos_dist, dim=2)[0] * V_mask).unsqueeze(2)
        # TODO Mask may not be needed. they are not selected. i.e not bpp anyway

        if self._embed_pos:
            # [m, max_num_doms, 1 + d_tag + d_text+d_class + 1 + 2 + 2(13)]
            dom_embeds = torch.cat(
                    (top_tokens, tag_embeds, classes_embeds,
                        alignments, focus_encodes, tampered_encodes), dim=2)
        else:
            # [m, max_num_doms, d_tag + d_text + 1 + 2 + 2(13)]
            dom_embeds = torch.cat(
                    (tag_embeds, classes_embeds, alignments,
                        focus_encodes, tampered_encodes), dim=2)
        return h_goal, pos_goal_embeds, goal_mask, dom_embeds, V_mask

    @property
    def E_dom_dim(self):
        return self._E_dom_dim

    @property
    def track_info(self):
        return {"Module":
                   {
                       "E_tag":self._E_tag.track_info,
                       "E_text": self._E_text.track_info,
                       "E_classes": self._E_classes.track_info
                   }
               }

    def debug_h(self, x):
        dom_embeds = self([x])
        return {"h": dom_embeds.squeeze(0)}, {}

    @property
    def text_dim(self):
        return self._E_text.embedding_dim

    def _oov_rand_mask(
            self, text_embeds, text_oov_mask, text_oov_ids,
            goal_embeds, goal_oov_mask, goal_oov_ids):
        """

        goal_oov_mask [m, max_num_goals]
        goal_oov_ids  [m, max_num_goals] -> [m, max_num_goals, E_text_dim]
        rand_embeds.gather(...) -> [m, max_num_goals, E_text_dim]
        """
        num_ids = self._max_num_doms + self._max_num_goal_tokens
        rand_embeds = torch.randn((num_ids, self._E_text_dim), device=self._oov_rand_device)
        # text embeds flatten [m*max_num_doms, E_text]
        m = len(text_embeds)
        text_embeds = text_embeds.view(m*self._max_num_doms, self._E_text_dim)

        # goal embeds flatten [m*max_num_goals, E_text]
        goal_embeds = goal_embeds.view(m*self._max_num_goal_tokens, self._E_text_dim)

        if len(text_oov_ids) > 0:
            # ipdb.set_trace()
            text_embeds[text_oov_mask==1] = rand_embeds[text_oov_ids]
        if len(goal_oov_ids) > 0:
            # ipdb.set_trace()
            goal_embeds[goal_oov_mask==1] = rand_embeds[goal_oov_ids]
        text_embeds = text_embeds.view(m, self._max_num_doms, self._E_text_dim)
        goal_embeds = goal_embeds.view(m, self._max_num_goal_tokens, self._E_text_dim)
        return text_embeds, goal_embeds




def batch_pad_(batch_tokens, pad_token, device, max_num_tokens=None):
    """
    In-place
    """
    if max_num_tokens is None:
        max_num_tokens = max(len(sub_tokens) for sub_tokens in batch_tokens)
    mask = torch.ones(len(batch_tokens), max_num_tokens, device=device)
    for sub_tokens, submask in zip(batch_tokens, mask):
        if len(sub_tokens) < max_num_tokens:
            submask[len(sub_tokens):max_num_tokens] = 0.
            sub_tokens.extend([pad_token]*(max_num_tokens - len(sub_tokens)))
    return mask

