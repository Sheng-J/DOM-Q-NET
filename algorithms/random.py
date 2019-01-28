import numpy as np
import torch
import random
import math
import ipdb


def epsilon_greedy_wrap(greedy_f, eps_config, device=torch.device("cpu")):
    action_space_f = eps_config.action_space_f
    past_eps = eps_config.eps_schedule_f(0)
    print("Initial epsilon: %.3f" % past_eps)

    random_f = create_random_f(action_space_f, device)
    def epsilon_policy(s_t, t, raw_s_t):
        """
        s_t is the tensor format ready for net to use
        """
        nonlocal past_eps
        sample = random.random()
        eps = eps_config.eps_schedule_f(t)
        if (past_eps - eps) >= eps_config.eps_print_diff:
            past_eps = eps
            print("Current epsilon: %.3f" % eps)

        if sample > eps:
            return greedy_f(s_t, t, raw_s_t)
        else:
            a_t = random_f(raw_s_t)
            # [1, num_actions]
            # q_vals = q_net(s_t)
            # a_t_idx = torch.tensor([[a_t]], device=device)
            # q_t = q_vals.gather(dim=1, index=a_t_idx).detach().item()
            # return a_t, q_t, q_vals
            return a_t, None, None
    return epsilon_policy


def create_random_f(action_space_f, device):
    def random_policy(raw_s_t):
        action_space = action_space_f(raw_s_t)
        # a_t = random.randrange(action_space)
        a_t = torch.tensor(
                [[random.randrange(action_space)]],
                device=device, dtype=torch.long
                )
        return a_t
    return random_policy


# TODO SOME PROBLEMS WITH NONE, NONE






