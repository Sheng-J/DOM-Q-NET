import time
import os
import torch
import torch.nn as nn
import configs
from algorithms import qlearn
from dstructs import replay, prio_replay
from utils import track, trackX, schedule
import ipdb
from entries.template import create_build_f, create_env_f, create_action_space_f
from actors import dqn_actor


NUM_CONSECUTIVE = 10


def main(res_dir, settings, hparams_list, paths_list, prints_dict):
    paths = paths_list[0]
    buffer_device = torch.device(settings.get("buffer_device", "cuda:0"))
    device = torch.device(settings.get("batch_device", "cuda:0"))
    saved_path = paths["saved_path"]
    ckpt = torch.load(saved_path)
    nn_hs, qlearn_hs, replay_hs, other_hs = ckpt["hparams_list"]
    V_tag, V_text, V_class = ckpt["V_tag"], ckpt["V_text"], ckpt["V_class"]

    build_net_f, save_dict, common_track_f = create_build_f(
            nn_hs, qlearn_hs, other_hs, prints_dict, None, V_text, V_tag, V_class
            )
    q_net, net_track_f = build_net_f(buffer_device, device)
    q_net.load_state_dict(ckpt["net"])
    q_net.eval()

    # Configure env f 
    env_f = create_env_f(nn_hs, qlearn_hs, other_hs, settings)
    q = input("START")
    i = 0

    if settings.get("multi_env", False):
        env_fs = env_f
        actors = []
        done_actors = []
        rewards = []
        for env_f in env_fs:
            actor = dqn_actor.Actor(
                    env_f, None, q_net,
                    None, None, None,
                    qlearn_hs["max_step_per_epi"], None, None, buffer_device, device
                    )
            actors.append(actor) 
            done_actors.append(False)
            rewards.append(None)
        num_actors = len(actors)
        while True:
            for i in range(num_actors):
                if not done_actors[i]:
                    reward, done = actors[i].just_act()
                    done_actors[i] = done
                    rewards[i] = reward
            time.sleep(2.0)
            all_done = True
            for done in done_actors:
                if not done:
                    all_done = False
                    break
            if all_done:
                reward_str = "_".join([str(int(reward)) for reward in rewards])
                if i < NUM_CONSECUTIVE:
                    for i in range(num_actors):
                        actors[i].reset()
                        done_actors[i] = False
                        rewards[i] = None
                    time.sleep(2.0)
                else:
                    q = input("Reward=%s, Press any (q quit) to continue..."%reward_str)
                    if q == "q":
                        break
                    else:
                        for i in range(num_actors):
                            actors[i].reset()
                            done_actors[i] = False
                            rewards[i] = None
                        time.sleep(2.0)
                i+=1

    else:
        actor = dqn_actor.Actor(
                env_f, None, q_net,
                None, None, None,
                qlearn_hs["max_step_per_epi"], None, None, device
                )
        q = input("START")
        while True:
            reward, done = actor.just_act()
            time.sleep(2)
            if done:
                actor.reset()
                time.sleep(2.0)
                #q = input("Reward=%d, Press any (q quit) to continue..."%int(reward))
                #if q == "q":
                #    break
                #else:
                #    actor.reset()
                #    time.sleep(2.0)



