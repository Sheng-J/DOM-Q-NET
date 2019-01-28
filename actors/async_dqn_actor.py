import torch
import multiprocessing as mp
from algorithms import random
from actors.dqn_actor import create_greedy_f, prep_one_trans_cpu_tensors


ACTOR_NEEDS_UPDATE = 0
ACTOR_DONE = 1


class ActorProcess(mp.Process):
    """
    BLOCK:
      [1] Target needs to be updated.
    """
    def __init__(
            self, i, init_step, init_epi, env_f, q_net_f, n_steps, replay_queue, net_conn,
             T_update, T_total, eps_config, gamma, loc_buff_size,
            max_step_per_epi, tracker
            ):
        super(ActorProcess, self).__init__()
        self._id = i
        self._init_step = init_step
        self._init_epi = init_epi
        self._n_steps = n_steps
        self._replay_queue = replay_queue
        self._net_conn = net_conn
        self._env_f = env_f
        self._q_net_f = q_net_f
        self._T_total = T_total
        self._T_update = T_update
        self._eps_config = eps_config
        self._gamma = gamma
        self._loc_buff_size = loc_buff_size
        self._max_step_per_epi = max_step_per_epi
        self._tracker = tracker

    def run(self):
        print("Actor%d Started"%self._id)
        t, epi = self._init_step, self._init_epi
        env, tracker = self._env_f(), self._tracker
        q_net = self._q_net_f()
        greedy_f = create_greedy_f(q_net)
        action_f = random.epsilon_greedy_wrap(q_net, greedy_f, self._eps_config) 

        raw_s_t = env.reset()
        s_t = q_net.prep(raw_s_t)
        s_t_tensors = prep_cpu_tensors(s_t)
        n_trans = []
        local_buffer = []
        t_ = t
        while t < self._T_total:
            # ndarray
            # 1, 1,         [1, num_actions]
            a_t, q_t, q_vals = action_f(s_t_tensors, t, raw_s_t)
            raw_s_t_1, r_t, done, _ = env.step(a_t)
            if not done:
                s_t_1 = q_net.prep(raw_s_t_1)
                s_t_1_tensors = prep_cpu_tensors(s_t)
            else:
                s_t_1 = None
                s_t_1_tensors = None
            n_trans.append((s_t, a_t, r_t, q_t))  # s, a, r all python values
            t += 1
            raw_s_t = raw_s_t_1
            s_t = s_t_1
            s_t_tensors = s_t_1_tensors

            if (t-t_) == self._n_steps or done or env.epi_step > self._max_step_per_epi:
                #
                # Regardless of cond, get <N steps trans
                #
                if not done:
                    q_vals = q_net(s_t_tensors)
                    q_t_n = q_vals.max(dim=1)[0].detach().item()
                else:
                    q_t_n = 0

                r_t_n = 0
                s_t_n = s_t_1
                while len(n_trans) > 0:
                    # backwards from t_n to 1
                    s_t_, a_t_, r_t_, q_t = n_trans.pop()
                    r_t_n = r_t + self._gamma * r_t_n
                    td_err = q_t - (r_t_n + q_t_n)
                    local_buffer.append(((s_t_, a_t_, r_t_n, s_t_n), td_err))
                t_ = t
                if len(local_buffer) > self._loc_buff_size:
                    self._replay_queue.put(local_buffer)
                    local_buffer = []
                
                #
                # Handle done or max step epi
                #
                if env.epi_step > b_config.max_step_per_epi:
                    print("Episode prematurely teminated")


                if done or env.epi_step>self._max_step_per_epi:
                    tracker.track_t("total_epi", t, epi)
                    tracker.tracks(
                            ["reward", "step", "eps"],
                            t,
                            [env.epi_reward, env.epi_step, eps_config.eps_schedule_f(t)])
                    epi += 1
                    # if done, override start s_t
                    raw_s_t = env.reset()
                    s_t = q_net.prep(raw_s_t)
                    s_t_tensors = prep_cpu_tensors(s_t)

            if t % self._T_update == 0:
                print("Actor%d blocked for update"%self._id)
                self._net_conn.send(ACTOR_NEEDS_UPDATE)
                q_state_dict = self._net_conn.recv()
                q_net.load_state_dict(q_state_dict)
                print("Actor%d updated"%self._id)
        self._net_conn.send(ACTOR_DONE)



