import os
import torch
import torch.nn as nn
import configs
from algorithms import qlearn
from dstructs import replay, prio_replay
from utils import track, trackX, schedule
import ipdb


def create_q_entry(create_build_f, create_env_f, create_action_space_f):
    """
    create_build_f(Qnet, nn_hs, qlearn_hs, other_hs, ckpt)
    create_env_f(nn_hs, qlearn_hs, other_hs, settings)
    """
    def main(res_dir, settings, hparams_list, paths_list, prints_dict):
        buffer_device = torch.device(settings.get("buffer_device", "cuda:0"))
        batch_device = torch.device(settings.get("batch_device", "cuda:0"))
        saved_path = os.path.join(res_dir, "ckpt")
        print("Saving result under %s" % saved_path) 

        ckpt_exists = os.path.exists(saved_path)

        if not ckpt_exists:
            ckpt = None
            init_epi, init_step = 0, 0
            nn_hs, qlearn_hs, replay_hs = hparams_list[0], hparams_list[1], hparams_list[2]
            T_total = qlearn_hs["T_total"]
            other_hs = {}
            for d in hparams_list[3:]:
                for key, val in d.items():
                    other_hs[key] = val
            # SETUP Track Ts
            io_T, last_n = settings["io_T"], settings["print_last_n_avg"]
            replay_track_T = settings["T_replay_track"]
            # Print key,vals Period
            io_T_dict = {
                    "reward": io_T, "step": io_T, "TD_err": io_T,
                    "cross-entropy": io_T,"total_epi": io_T, "eps": io_T,
                    "avg_times_in_buffer_per_tran": replay_track_T,
                    "avg_times_sampled_per_tran": replay_track_T, 
                    "avg_times_sampled_per_pos_tran": replay_track_T,
                    "avg_times_sampled_per_neg_tran": replay_track_T, 
                    "num_pos_trans": replay_track_T, 
                    "num_neg_trans": replay_track_T,
                    "frac_pos_trans": replay_track_T, 
                    "avg_num_pos_trans_per_batch": replay_track_T, 
                    "avg_num_neg_trans_per_batch": replay_track_T,
                    "avg_num_unique_pos_trans_per_batch": replay_track_T,
                    "avg_num_unique_neg_trans_per_batch": replay_track_T,
                    "avg_pos_trans_err": replay_track_T, 
                    "avg_neg_trans_err": replay_track_T
                    }
            # Last n avg
            last_n_avg_dict = {
                    "reward": last_n, "step": last_n, "eps": last_n}
            decimal_dict = {
                    "reward": 2, "step": 2, "total_epi": 2, "TD_err": 4,
                    "cross-entropy": 3,"total_epi": 2, "eps": 2,
                    "avg_times_in_buffer_per_tran": 2,
                    "avg_times_sampled_per_tran": 2,
                    "avg_times_sampled_per_pos_tran": 2,
                    "avg_times_sampled_per_neg_tran": 2,
                    "num_pos_trans": 2,
                    "num_neg_trans": 2,
                    "frac_pos_trans": 2,
                    "avg_num_pos_trans_per_batch": 2,
                    "avg_num_neg_trans_per_batch": 2,
                    "avg_num_unique_pos_trans_per_batch": 2,
                    "avg_num_unique_neg_trans_per_batch": 2,
                    "avg_pos_trans_err": 4,
                    "avg_neg_trans_err": 4
                    }
            # TFboard track period
            track_T_dict = {
                    "TD_err":settings["err_T_track"], "cross-entropy":settings["err_T_track"],
                    "avg_times_in_buffer_per_tran": replay_track_T,
                    "avg_times_sampled_per_tran": replay_track_T,
                    "avg_times_sampled_per_pos_tran": replay_track_T,
                    "avg_times_sampled_per_neg_tran": replay_track_T,
                    "num_pos_trans": replay_track_T, 
                    "num_neg_trans": replay_track_T,
                    "frac_pos_trans": replay_track_T,
                    "avg_num_pos_trans_per_batch": replay_track_T,
                    "avg_num_neg_trans_per_batch": replay_track_T,
                    "avg_num_unique_pos_trans_per_batch": replay_track_T,
                    "avg_num_unique_neg_trans_per_batch": replay_track_T,
                    "avg_pos_trans_err": replay_track_T,
                    "avg_neg_trans_err": replay_track_T
                    }
            if replay_hs["replay_type"] == "default":
                print("Use default replay buffer")
                replay_buffer = replay.ReplayBuffer(replay_hs["capacity"])
            elif replay_hs["replay_type"] == "td_err_prioritized":
                print("NEW prioritized replay buffer")
                replay_buffer = prio_replay.Buffer(replay_hs["capacity"])
            else:
                raise ValueError("Not implemented")
            replay_track_f = replay.create_track_f(replay_buffer, replay_track_T)
            tracker = track.Tracker(
                    res_dir, io_T_dict, last_n_avg_dict,
                    settings["export_T"], decimal_dict,
                    track_T_dict
                    )
        else:
            ckpt = torch.load(saved_path)
            init_epi, init_step = ckpt["init_epi"], ckpt["init_step"]
            nn_hs, qlearn_hs, replay_hs, other_hs = ckpt["hparams_list"]
            T_total = qlearn_hs["T_total"]
            replay_buffer = ckpt["replay_buffer"]
            tracker = ckpt["tracker"]

        T_exploration = qlearn_hs['T_exploration']
        qlearn_hs["use_c51"] = qlearn_hs.get("use_c51", False)

        # Net creation & save
        # build_net_f loads from ckpt if exists
        ckpt_path = paths_list[1] if len(paths_list) >= 2 else None
        build_net_f, save_dict, common_track_f = create_build_f(
                nn_hs, qlearn_hs, other_hs, prints_dict, ckpt_path, ckpt
                )
        q_net, net_track_f = build_net_f(buffer_device, batch_device)
        save_dict["net"] = q_net.state_dict()
        save_dict["hparams_list"] = (nn_hs, qlearn_hs, replay_hs, other_hs)

        if settings.get('save_ckpt', False):
            def save_f(step=None):
                if step is None:
                    torch.save(save_dict, saved_path)
                elif step % settings["T_ckpt"] == 0:
                    torch.save(save_dict, saved_path+str(step))
                    print("Saved at %d" % step)
        else:
            def save_f(step=None):
                return
        tgt_net, _ = build_net_f(buffer_device, batch_device)
        tgt_net.load_state_dict(q_net.state_dict())

        if qlearn_hs["opt_type"] == "rms_prop":
            optimizer = torch.optim.RMSprop(q_net.parameters(), lr=qlearn_hs["lr"])
        else:
            optimizer = torch.optim.Adam([para for para in q_net.parameters() if para.requires_grad], lr=qlearn_hs["lr"])
            # optimizer = torch.optim.Adam(q_net.parameters(), lr=qlearn_hs["lr"])
        #if ckpt_exists:
        #    optimizer.load_state_dict(ckpt["optimizer"])
        # TF Board setup
        if len(paths_list) >= 1 and "plot_path" in paths_list[0]:
            trackerX = trackX.TrackerX(
                    paths_list[0]["plot_path"],
                    res_dir, tracker
                    )
        else:
            trackerX = tracker

        t_config = configs.TrainConfig(
                qlearn_hs["batch_size"], {"q_net":q_net, "tgt_net":tgt_net},
                optimizer, qlearn_hs["grad_clip"], qlearn_hs["gamma"], init_step,
                T_total, init_epi, buffer_device, batch_device, qlearn_hs["max_step_per_epi"], save_f,
                )
        t_config.print_info()

        env = create_env_f(nn_hs, qlearn_hs, other_hs, settings)

        action_space_f = create_action_space_f(nn_hs, qlearn_hs, other_hs, settings)
        save_f()
        if "n_actors" in other_hs:
            print("Async Q learn")
            eps_config = configs.EpsilonGreedyConfig(
                    None, action_space_f,
                    settings.get('eps_diff_print', 0.01), T_exploration
                    )
            async_qlearn.async_train(
                    t_config, env_f, replay_buffer, qlearn_hs["n_steps"],
                    qlearn_hs.get("use_ddqn", True),
                    qlearn_hs["T_tgt_update"], build_net_f, other_hs["n_actors"], other_hs["max_push_per_sample"],
                    other_hs["T_actor_update"], other_hs["actor_loc_buffer_size"],
                    eps_config, trackerX
                    )
        else:
            if not q_net.use_noisylayers:
                eps_scheduler = schedule.create_linear_schedule(
                        qlearn_hs["eps_decay_frac"] * T_total,
                        y_0=qlearn_hs["eps_start"],
                        y_T=qlearn_hs["eps_end"],
                        offset_val=1.0,
                        offset_t=T_exploration
                        )
                eps_config = configs.EpsilonGreedyConfig(
                        eps_scheduler, action_space_f,
                        settings.get('eps_diff_print', 0.01), T_exploration
                        )
            else:
                eps_config = None

            if settings.get("multi_env", False):
                train_f = qlearn.multitask_train
                trackerX = []
                for i in range(len(settings["env"])):
                    env_name = settings["env"][i]
                    tracker_ = track.Tracker(
                            res_dir + "_" + env_name, io_T_dict, last_n_avg_dict,
                            settings["export_T"], decimal_dict,
                            track_T_dict, env_name
                            )
                    trackerX_ = trackX.TrackerX(
                            paths_list[0]["plot_path"+str(i+1)],
                            res_dir+str(i+1), tracker_
                            )
                    trackerX.append(trackerX_)
            else:
                train_f = qlearn.train
            q_net.train()
            tgt_net.train()
            train_f(
                    t_config, env, replay_buffer, qlearn_hs["n_steps"],
                    qlearn_hs.get("use_ddqn", True),
                    qlearn_hs["use_c51"], 
                    qlearn_hs["T_tgt_update"],
                    T_exploration,
                    qlearn_hs["T_update"],
                    eps_config, trackerX,
                    [common_track_f, net_track_f, replay_track_f],
                    V_min=qlearn_hs.get("V_min"), V_max=qlearn_hs.get("V_max"),
                    num_atoms=qlearn_hs.get("num_atoms")
                    )

        save_f()
    return main




