import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ipdb
import os
import numpy as np
import time
from pprint import pprint
from actors.dqn_actor import prep_one_trans_device_tensors


class InteractiveEnvironment:
    def __init__(self, env):
        self._env = env

    def run_episode(self):
        print("-------------------")
        self._env.reset_show()
        is_done = self._env.is_done
        while (not is_done) and (self._env.epi_step < 5):
            A_t = int(input("Select an action: "))
            is_done = self.step(A_t)
        print("-------------------\n\n")

    def step(self, A_t):
        print(self._env.is_done)
        if self._env.is_done:
            print("TIMED OUT")
            return True
        else:
            res_tuple = self._env.step_show(A_t)
            if res_tuple[2]:
                return True
            return False


class AgentEnvironment:
    def __init__(self, env, net, action_f, device, save_dir):
        self._env = env
        self._net = net
        self._action_f = action_f
        self._device = device
        self._save_dir = save_dir

    def run_episode(self, i):
        print("-------------------")
        x = self._env.reset()
        is_done = self._env.is_done
        while (not is_done) :
            S_t = self._net.prep(x)
            S_t_tensors = prep_one_trans_device_tensors(S_t, self._device)

            res = self._net.get_attn_weights(S_t_tensors)
            self.save_attn(i, self._env.epi_step, res)

            A_t, q_dom_max, q_vals = self._action_f(S_t_tensors)
            doms_info, leaves_info = self._net.debug_h(S_t)

            leaves_info["Q"] = q_vals.squeeze(0).cpu().numpy()
            is_done, x = self.step(A_t, doms_info, leaves_info)
        print("-------------------\n\n")

    def step(self, A_t, doms_info, leaves_info):
        if self._env.is_done or (self._env.epi_step >= 3):
            print("TIMED OUT")
            return True, None
        else:
            res_tuple = self._env.debug_step(A_t.item(), doms_info, leaves_info)
            if res_tuple[2]:
                return True, res_tuple[0]
            return False, res_tuple[0]


    def save_attn(self, i, step, res):
        attn_w, A, labels = res
        V = 14
        labels = labels[:V]
        labels.insert(0, '_')
        f_label = "labels-eps%d-step%d"%(i, step)

        f_name2 = "A-eps%d-step%d"%(i, step)
        A = A.squeeze(0).cpu().detach().numpy()[:V, :V]
        saveAttention(labels, A, os.path.join(self._save_dir, f_name2))
        # np.save(os.path.join(self._save_dir, f_name2), A)
        for prop_j in range(len(attn_w)):
            f_name = "P-attn-eps%d-step%d-prop%d"%(i, step, prop_j)
            attn = attn_w[prop_j].squeeze(0).cpu().detach().numpy()[:V, :V]
            assert attn.shape == A.shape
            saveAttention(labels, attn, os.path.join(self._save_dir, f_name))
            # np.save(os.path.join(self._save_dir, f_name), attn)


def saveAttention(labels, attentions, save_dir):
    global num
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, aspect='equal')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.savefig(save_dir)
    ax.clear()
    plt.cla()
    plt.clf()
    plt.close()
    # time.sleep(1)



