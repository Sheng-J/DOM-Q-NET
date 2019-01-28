import os
import torch
import torch.nn as nn
import plp.vocab as pvocab
import configs
from models.miniwob.embed import DomLeavesEmbedding
from miniwob.env import MiniWoBEnvironment
from miniwob import custom
from models import rnn, embed, graph
from algorithms import qlearn
from dstructs import replay
from utils import track, trackX, schedule
from models import dom_qnet
import ipdb
import numpy as np


def create_build_f(nn_hs, qlearn_hs, other_hs, prints_dict, ckpt_path=None, V_text=None, V_tag=None, V_class=None):
    if V_tag is None:
        V_tag = pvocab.LazyVocab("tag", nn_hs["max_tag_vocab_size"], nn_hs["allow_unk"])
    if V_text is None:
        if ckpt_path is not None:
            pretrain_ckpt = torch.load(ckpt_path)
            V_text = pretrain_ckpt["V_text"]
        else:
            V_text = pvocab.LazyVocab("text", nn_hs["max_text_vocab_size"], nn_hs["allow_text_unk"])
    if V_class is None:
        V_class = pvocab.LazyVocab("class", nn_hs["max_class_vocab_size"], nn_hs["allow_unk"])
    def build_net_f(buffer_device, batch_device):
        ##
        # Attr Embeddings construction
        ##
        E_text1D = embed.Embedding(V_text, nn_hs["E_text_dim"])
        if ckpt_path is not None:
            print("Use pretrained")
            E_text1D._E.load_state_dict(pretrain_ckpt["E_state_dict"])
            if nn_hs.get('fix_E_text'):
                for param in E_text1D.parameters():
                    param.requires_grad = False

        E_text_attr = E_text1D

        E_goal = rnn.EmbedNet(
                 nn_hs["E_text_dim"], other_hs["goal_max_num_tokens"], E_text1D, batch_device
                 )
        E_tag = embed.Embedding(V_tag, nn_hs["E_tag_dim"])
        E_class = embed.Embedding(V_class, nn_hs["E_class_dim"]) 
        ##
        # E_x construction 
        ##
        E_x = DomLeavesEmbedding(E_tag, E_text_attr, E_class, E_goal, other_hs["V_size"], batch_device, nn_hs.get("embed_top", False))
        ##
        # All nodes Graph Embeddings construction
        ##
        ggnn_net = graph.GgnnUndirectedEmbed(
                other_hs["n_prop_steps"], E_x.E_dom_dim,
                other_hs["V_size"], 1, other_hs.get("aggr_type")
                )
        def net_track_f(trackerX, t):
            if ("T_print_E" in prints_dict) and (t % prints_dict["T_print_E"] == 0):
                pass
                #trackerX.add_E(
                #        E_text_attr.weight.detach().cpu()[:len(V_text)],
                #        t, V_text.labels, tag=("E_text_"+prints_dict["res_dir"])
                #        )
                #trackerX.add_E(
                #        E_tag.weight.detach().cpu()[:len(V_tag)],
                #        t, V_tag.labels, tag=("E_tag_"+prints_dict["res_dir"])
                #        )
                #trackerX.add_E(
                #        E_class.weight.detach().cpu()[:len(V_class)],
                #        t, V_class.labels, tag=("E_class_"+prints_dict["res_dir"])
                #        )
                #cos_dist_mat = E_text_attr.get_E_cos_dist_mat(len(V_text))
                #fig = track.create_mat_figure(cos_dist_mat, V_text.labels)
                #trackerX.add_img(("cos_E_text_"+prints_dict["res_dir"]), np.array(fig.canvas.renderer._renderer), t)
        ##
        # C51 vs DQN - ret leaves pred
        ##
        if qlearn_hs["use_c51"]:
            net = dom_qnet.Qnet(
                    E_x, ggnn_net, other_hs["max_num_doms"],
                    nn_hs["fc_dim"], num_atoms=qlearn_hs["num_atoms"],
                    use_c51=True,
                    dueling_type=qlearn_hs.get("dueling_type"),
                    use_noisylayers=qlearn_hs.get("use_noisylayers", False)
                    )
        else:
            net = dom_qnet.Qnet(
                    E_x, ggnn_net, other_hs["max_num_doms"],
                    nn_hs["fc_dim"],
                    use_c51=False,
                    dueling_type=qlearn_hs.get("dueling_type"),
                    use_noisylayers=qlearn_hs.get("use_noisylayers", False),
                    use_goal_attn=other_hs["use_goal_attn"],
                    use_goal_cat=other_hs["use_goal_cat"],
                    use_local=other_hs["use_local"],
                    use_neighbor=other_hs["use_neighbor"],
                    use_global=other_hs["use_global"]
                    )
        net = net.to(batch_device)
        return net, net_track_f
    def common_track_f(tracker, t):
        if ("T_print_Vocab" in prints_dict) and (t % prints_dict["T_print_Vocab"] == 0):
            pass
            #track_keys = ["V_text_size", "V_tag_size", "V_class_size"]
            #track_vals = [len(V_text), len(V_tag), len(V_class)]
            #tracker.tracks(track_keys, t, track_vals)
    save_dict = {
            "V_tag": V_tag, "V_text": V_text, "V_class": V_class
            }
    attr_vocabs = {
            "tag": V_tag, "text": V_text, "classes": V_class
            }
    other_hs["attr_vocabs"] = attr_vocabs
    return build_net_f, save_dict, common_track_f


def create_env_f(nn_hs, qlearn_hs, other_hs, settings):
    customizer = custom.create_customizer(
        settings.get("custom_mode"),
        other_hs["attr_vocabs"],
        )  
    if settings.get("multi_env", False):
        envs = [MiniWoBEnvironment(env_name, customizer)
            for env_name in settings["env"]]
        return envs
    else:
        return MiniWoBEnvironment(settings["env"], customizer)


def create_action_space_f(nn_hs, qlearn_hs, other_hs, settings):
    def action_space_f(x):
        return len(x[2]['ref'])
    return action_space_f






