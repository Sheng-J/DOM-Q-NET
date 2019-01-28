import multiprocessing as mp
import torch
import configs
from dstructs.async_replay import ReplayProcess
from actors.async_dqn_actor import ActorProcess
from learners.async_dqn_learner import steps_train
from utils import schedule
import ipdb


def async_train(
        t_config, env_f, replay_buffer, n_steps, T_tgt_update, build_net_f, n_actors,
        max_push_per_sample, T_actor_update, actor_loc_buff_size,
        eps_config_, tracker_, eps_=0.4, alpha=7.
        ):
    # actors
    actor_queues = []
    actor_processes = []
    actor_replay_queue = mp.Queue()
    learner_actor_conns = []
    for i in range(n_actors):
        actor_learner_conn, learner_actor_conn = mp.Pipe()
        learner_actor_conns.append(learner_actor_conn)
        # eps config setup
        # eps = eps_ ** (1 + alpha*i/(n_actors-1))
        eps = eps_
        # TODO FIX

        eps_scheduler = schedule.create_constant_schedule(1.0, eps_config_.t_exploration, eps)
        eps_config = configs.EpsilonGreedyConfig(
                eps_scheduler, eps_config_.action_space_f, eps_config_.eps_print_diff,
                eps_config_.t_exploration
                )
        tracker = tracker_.fork("act%d_eps%.2f"%(i, eps))
        actor_p = ActorProcess(
                i, t_config.init_step, t_config.init_epi, n_steps, actor_replay_queue,
                actor_learner_conn, env_f, build_net_f, T_actor_update, t_config.total_steps,
                eps_config, t_config.gamma, actor_loc_buff_size, t_config.max_step_per_epi,
                tracker
                )
        actor_queues.append(mp.Queue())
        actor_processes.append(actor_p)

    # async replay 
    learner_replay_conn, replay_learner_conn = mp.Pipe()
    replay_type = replay_buffer.sample_method
    replay_process = ReplayProcess(
            actor_queues, replay_learner_conn, replay_buffer,
            max_push_per_sample
            )

    # Start out actor processes and replay process
    for p in actor_processes:
        p.start()
    ipdb.set_trace()
    replay_process.start()

    # async learner main steps function
    steps_train(
            t_config, T_tgt_update, T_actor_update/10, learner_replay_conn,
            replay_type, learner_actor_conns
            )

    # TODO order
    for p in actor_processes:
        p.join()
    replay_process.join()

