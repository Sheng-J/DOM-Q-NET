import torch
from algorithms import random
from actors import c51_actor
import ipdb


def create_greedy_f(q_net, batch_device=None):
    def q_multiple_policy(s_t, t, raw_s_t):
        """
        argmax for each factorized action category
        """
        with torch.no_grad():
            a_tensor_ts = []
            a_ts = []
            qs = []
            q_vals = []
            # [1, num_actions]
            if batch_device is None:
                q_vals_list = q_net(s_t)
            else:
                s_t = [item.to(batch_device) for item in s_t]
                q_vals_list = q_net(s_t)

            for q_vals_ in q_vals_list:
                q_max_ = q_vals_.max(dim=1)
                a_t_ = q_max_[1].view(1, 1)
                a_tensor_ts.append(a_t_)
                a_ts.append(a_t_.detach().item())
                qs.append(q_max_[0].detach().item())
                q_vals.append(q_vals_)
            a_tensor_ts = torch.cat(a_tensor_ts, dim=1)
            return a_tensor_ts, a_ts, qs, q_vals
    return q_multiple_policy


def prep_one_trans_device_tensors(s_t, device):
    tensors = []
    s_t_2d, s_t_1d = s_t
    for item in s_t_2d:
        tensors.append(torch.tensor([item], device=device))
    for item in s_t_1d:
        tensors.append(torch.tensor(item, device=device))
    # ipdb.set_trace()
    #for item in s_t:
    #    try:
    #        tensors.append(torch.tensor([item], device=device))
    #    except:
    #        ipdb.set_trace()
    #        print()
    return tensors


class Actor:
    """
    RL algorithm dependent actor to the environment,
    env: func that returns environment with "reset" and "step"
    greedy_f - only diff between DistQ and Q actors. If epsilon greedy config
               is provided, this function is wrapped with epison greedy.  
               o.w. this init expects a qnet with noisy layers
    n_steps - num consecutive actions before return.
              i.e. Network is not updated withing those number of actions
    q_net - Core differentiable Q NN object 
    gamma - This class also computes nsteps return reward, so gamma required

    max_step_per_epi - to avoid infinite loop.  reward will be zero with such termination
    tracker - is the tracking object for storing experiment results etc.
    track_fs - optional list with tracking functions to run 
               (the ones that cannot be encapsulated in tracker)
    eps_config - epsilon greedy configuration.  supply "None" if using noisy layer
    device - device to place the tensor
    T_total - total number of frames to run

    Punchline: The actor class stores the environment to which it acts on.
    The separation is not necessary as this will require supplying env as an argument
    at almost all methods, but environment is almost purely used by actor.
    """
    def __init__(
            self, env, n_steps, q_net, T_total, eps_config,
            gamma, max_step_per_epi, tracker, track_fs, buffer_device, batch_device
            ):
        # attrs pointing to same addr
        self._env = env
        self._T_total = T_total
        self._q_net = q_net
        self._device = buffer_device
        self._max_step_per_epi = max_step_per_epi
        self._gamma = gamma
        self._tracker = tracker
        self._track_fs = track_fs
        self._eps_config = eps_config

        # updated after each "__call__"
        self._raw_s_t = self._env.reset() 
        self._s_t = q_net.prep(self._raw_s_t)
        self._s_t_tensors = prep_one_trans_device_tensors(self._s_t, buffer_device)
        self._t = 0
        self._epi = 0
        
        # main func
        if batch_device != buffer_device:
            greedy_f = create_greedy_f(q_net, batch_device)
        else:
            greedy_f = create_greedy_f(q_net)

        if self._eps_config is not None:
            self._action_f = random.epsilon_greedy_wrap(greedy_f, eps_config, device)
        else:
            self._action_f = greedy_f

        self._n_trans = []
        self._n_steps = n_steps

    def update_greedy_f(self, create_greedy_f, *extra_args):
        greedy_f = create_greedy_f(self._q_net, *extra_args)
        if self._eps_config is not None:
            self._action_f = random.epsilon_greedy_wrap(greedy_f, self._eps_config, self._device)
        else:
            self._action_f = greedy_f

    def __call__(self):
        """
        This executes the actor n times to the environment and return the 
        n steps transitions. 
        raw_s_t is unprocessed state returned by the environment
        s_t is processed state for conformed representation
        s_t_tensors is processed state placed on the device
        """
        raw_s_t, s_t, s_t_tensors = self._raw_s_t, self._s_t, self._s_t_tensors
        a_t_tensors, a_t, q_t, q_vals = self._action_f(s_t_tensors, self._t, raw_s_t)
        raw_s_t_1, raw_r_t, done, _ = self._env.step(a_t)
        r_t = torch.tensor([raw_r_t], dtype=torch.float, device=self._device)
        self._n_trans.append(
                (s_t_tensors, a_t_tensors, r_t, q_t)
                )
        # track funtions
        for track_f in self._track_fs:
            track_f(self._tracker, self._t)
        self._t += 1
        s_t_1, s_t_1_tensors = self._get_s_t_1(raw_s_t_1, done)

        done_cond = done or self._env.epi_step > self._max_step_per_epi

        # Generate nstep trans result
        if len(self._n_trans) == self._n_steps:
            r_t_n = self._aggr_nsteps_return()
            s_0, a_0, _, __ = self._n_trans.pop(0)
            n_step_trans = [(s_0, a_0, r_t_n, s_t_1_tensors)]
        else:
            n_step_trans = []

        ###
        # Handle done or max step epi
        ###
        if self._env.epi_step > self._max_step_per_epi:
            print("Episode forcefully terminated")
        if done_cond:
            if self._eps_config is not None:
                track_keys = ["reward", "step", "eps", "total_epi"]
                track_vals = [self._env.epi_reward, self._env.epi_step, self._eps_config.eps_schedule_f(self._t), self._epi]
            else:
                track_keys = ["reward", "step", "total_epi"]
                track_vals = [self._env.epi_reward, self._env.epi_step, self._epi]
            self._tracker.tracks(track_keys, self._t, track_vals)
            self._epi += 1
            # Reste s_t if done or max step
            while len(self._n_trans) > 0:
                r_t_n = self._aggr_nsteps_return()
                s_0, a_0, _, __ = self._n_trans.pop(0)
                n_step_trans.append((s_0, a_0, r_t_n, s_t_1_tensors))
            self._raw_s_t = self._env.reset()
            self._s_t = self._q_net.prep(self._raw_s_t)
            self._s_t_tensors = prep_one_trans_device_tensors(self._s_t, self._device)
        else:
            self._raw_s_t, self._s_t, self._s_t_tensors = raw_s_t_1, s_t_1, s_t_1_tensors 
        return n_step_trans, self._t

    def _get_s_t_1(self, raw_s_t_1, done):
        if not done:
            s_t_1 = self._q_net.prep(raw_s_t_1)
            s_t_1_tensors = prep_one_trans_device_tensors(s_t_1, self._device)
        else:
            s_t_1 = None
            s_t_1_tensors = None
        return s_t_1, s_t_1_tensors

    def _aggr_nsteps_return(self):
        r_t_n = 0
        j = len(self._n_trans) - 1 
        while j >= 0:
            # backwards from t_n to 1
            s_t_, a_t_, r_t_, q_t = self._n_trans[j]
            r_t_n = r_t_ + self._gamma * r_t_n
            j -= 1
        return r_t_n

    @property
    def T_done(self):
        return (self._t > self._T_total)

    @property
    def t(self):
        return self._t

    def just_act(self):
        raw_s_t, s_t, s_t_tensors = self._raw_s_t, self._s_t, self._s_t_tensors
        a_t_tensors, a_t, q_t, q_vals = self._action_f(s_t_tensors, self._t, raw_s_t)
        raw_s_t_1, raw_r_t, done, _ = self._env.step(a_t)
        s_t_1, s_t_1_tensors = self._get_s_t_1(raw_s_t_1, done)

        done_cond = done or self._env.epi_step > self._max_step_per_epi
        if self._env.epi_step > self._max_step_per_epi:
            print("Episode forcefully terminated")
        if done_cond:
            return self._env.epi_reward, True
        else:
            self._raw_s_t, self._s_t, self._s_t_tensors = raw_s_t_1, s_t_1, s_t_1_tensors 
            return self._env.epi_reward, False

    def reset(self):
        # Reste s_t if done or max step
        self._raw_s_t = self._env.reset()
        self._s_t = self._q_net.prep(self._raw_s_t)
        self._s_t_tensors = prep_one_trans_device_tensors(self._s_t, self._device)











