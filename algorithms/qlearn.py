import torch
import configs
from actors import dqn_actor, c51_actor
from learners import dqn_learner, c51_learner
from utils import schedule
import ipdb


def train(
        t_config, env, replay_buffer, n_steps, use_ddqn, use_c51, T_tgt_update,
        T_exploration, T_update, eps_config, tracker, track_fs, V_min=None, V_max=None, num_atoms=None,
        ):
    print("T_tgt_update=%d"%T_tgt_update)
    q_net, tgt_net = t_config.models["q_net"], t_config.models["tgt_net"]
    use_noisylayers = q_net.use_noisylayers
    actor = dqn_actor.Actor(
            env, n_steps, q_net,
            t_config.total_steps, eps_config, t_config.gamma,
            t_config.max_step_per_epi, tracker, track_fs, t_config.buffer_device,
            t_config.batch_device
            )
    if use_c51:
        pass
    else:
        sample_optimize_f = dqn_learner.create_sample_optimize_f(
                t_config, n_steps, use_ddqn, replay_buffer,
                tracker
                )
    train_started = False
    while not actor.T_done:
        # 0. Draw new noisy weights, if noisy mode
        if use_noisylayers and actor.t % T_update == 0:
            q_net.reset_noise()

        # 1. actor step (t +1 per update)
        nsteps_trans, t = actor()
        # 2. buffer step (Add multiple <= nsteps accum transitions when done)
        assert len(nsteps_trans) <= n_steps
        for nsteps_tran in nsteps_trans:
            replay_buffer.push(*nsteps_tran)
        # 3. learn step 
        if train_started or (len(replay_buffer) >= t_config.batch_size and t >= T_exploration):
            if not train_started:
                print("TRAIN STARTED")
                train_started = True
            if t % T_update == 0:
                sample_optimize_f(t)
            if t % T_tgt_update == 0:
                tgt_net.load_state_dict(q_net.state_dict())
        t_config.save_f(t)
    print("Actor DONE")


def multitask_train(
        t_config, envs, replay_buffer, n_steps, use_ddqn, use_c51, T_tgt_update,
        T_exploration, T_update, eps_config, trackers, track_fs,
        V_min=None, V_max=None, num_atoms=None,
        ):
    """
    One replay buffer for all transitions of different tasks
    Multiple actors act one -> 
    # actors update
    """
    print("T_tgt_update=%d"%T_tgt_update)
    q_net, tgt_net = t_config.models["q_net"], t_config.models["tgt_net"]
    use_noisylayers = q_net.use_noisylayers
    actors = []
    for env, tracker in zip(envs, trackers):
        actor_ = dqn_actor.Actor(
                env, n_steps, q_net,
                t_config.total_steps, eps_config, t_config.gamma,
                t_config.max_step_per_epi, tracker, track_fs, t_config.buffer_device,
                t_config.batch_device
                )
        actors.append(actor_)
        if use_c51:
            pass
        else:
            sample_optimize_f = dqn_learner.create_sample_optimize_f(
                    t_config, n_steps, use_ddqn, replay_buffer,
                    tracker
                    )
    num_actors = len(actors)
    train_started = False
    while not actors[-1].T_done:

        # 1. actor step (t +1 per update)
        for actor in actors:
            # 0. Draw new noisy weights, if noisy mode
            if use_noisylayers and actor.t % T_update == 0:
                q_net.reset_noise()
            nsteps_trans, t = actor()
            #if len(nsteps_trans) >0:
            #    print(nsteps_trans)
            #    ipdb.set_trace()
            # 2. buffer step (Add multiple <= nsteps accum transitions when done)
            assert len(nsteps_trans) <= n_steps
            for nsteps_tran in nsteps_trans:
                replay_buffer.push(*nsteps_tran)
            # 3. learn step 
            if train_started or (len(replay_buffer) >= t_config.batch_size and t >= T_exploration):
                if not train_started:
                    print("TRAIN STARTED")
                    train_started = True
                if t % T_update == 0:
                    sample_optimize_f(t)
                if t % T_tgt_update == 0:
                    tgt_net.load_state_dict(q_net.state_dict())
            t_config.save_f(t)
    print("Actor DONE")



