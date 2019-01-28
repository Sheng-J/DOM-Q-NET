import multiprocessing as mp
import torch
from dstructs.replay import Transition
from dstructs.async_replay import REPLAY_SAMPLE, REPLAY_EXIT, REPLAY_UPDATE
from actors.async_dqn_actor import ACTOR_NEEDS_UPDATE, ACTOR_DONE
from learners.dqn_learner import create_optimize_f


def wrap_prio_sample(optimize_f, replay_conn, m):
    def step(t):
        replay_conn.send((REPLAY_SAMPLE, (m, t)))
        batch_trans, ws, idxs = replay_conn.recv()
        TD_errs = optimize_f(batch_trans)
        replay_conn.send(REPLAY_UPDATE, (idxs, TD_errs))
    return step


def steps_train(t_config, T_tgt_update, T_act_update_check, replay_conn, replay_type, actor_conns):
    """
    async learner does not push to the buffer, only samples
    """
    step, epi, q_net, optimize_f = _setup(t_config)
    optimize_f = create_optimize_f(t_config)
    if replay_type == "td_err_prioritized":
        step_f = wrap_prio_sample(optimize_f, replay_conn)
    elif replay_type == "uniform":
        pass

    actors_active = [True for _ in range(len(actor_conns))]

    while True:
        step_f(step)
        step += 1

        if t % T_act_update_check == 0:
            for i, (actor_active, actor_conn) in enumerate(zip(actors_active, actor_conns)):
                if actors_active and actor_conn.poll():
                    op = actor_conn.recv()
                    if op == ACTOR_NEEDS_UPDATE:
                        actor_conn.send(q_net.state_dict())
                    elif op == ACTOR_DONE:
                        actors_active[i] = False
                    else:
                        raise ValueError()
            if True not in actors_active:
                break

        if t % T_tgt_update == 0:
            b_config.models['tgt_net'].load_state_dict(
                    q_net.state_dict()
                    )
    replay_conn.send(REPLAY_EXIT, None)


def _setup(t_config):
    init_step = t_config.init_step
    init_epi = t_config.init_epi
    q_net = t_config.models['q_net']
    optimize_f = create_optimize_step_f(t_config)
    return init_step, init_epi, q_net, optimize_f


