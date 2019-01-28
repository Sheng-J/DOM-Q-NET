import torch
import torch.nn.functional as F
import ipdb
from pprint import pprint
from dstructs import replay
from learners import dqn_learner
from actors import c51_actor


# DDQN NOT SUPPORTED ATM
def create_compute_P_Z_t_1_f(dist_q_net, dist_tgt_net, Z, num_atoms):
    print("NOT use ddqn")
    def compute_P_Z_t_1(s_t_1):
        P_Z, mask = dist_tgt_net(s_t_1) 
        P_Z = P_Z.detach()
        # [m', 1]
        a_batch = c51_actor.a_from_P_Z(P_Z, Z, mask).unsqueeze(1)
        # [m', 1, num_atoms]
        a_idxs = a_batch.expand(len(a_batch), num_atoms).unsqueeze(1)  
        # [m', num_atoms]
        P_Z_t_1 = P_Z.gather(dim=1, index=a_idxs).squeeze(1)
        return P_Z_t_1
    return compute_P_Z_t_1


def create_compute_kl_div_f(
        compute_P_Z_t_1, dist_q_net, m, n_steps, gamma, 
        num_atoms, V_min, V_max, Z, dz, device
        ):
    def compute_kl_div(batch, non_final_mask):
        non_final_mask_float = non_final_mask.type(torch.float)
        s_t, non_final_s_t_1 = batch.s_t, batch.s_t_1
        # log P_Z_all
        log_P_Z_t_all, __action_mask = dist_q_net(s_t, log=True)
        # [m, 1, num_atoms]
        a_idxs = batch.a_t.expand(m, num_atoms).unsqueeze(1) 
        # [m, num_atoms]
        log_P_Z_t = log_P_Z_t_all.gather(dim=1, index=a_idxs).squeeze(1)

        # Z_t_1
        # [m, num_atoms]
        P_Z_t_1 = torch.zeros((m, num_atoms), device=device) 
        # Set P(Z=0)=1.0 for terminated S_t_1
        # -float(V_min)/float(dz) gives zero index
        P_Z_t_1[:, int(-float(V_min)/float(dz))] = 1.0  
        if len(batch.s_t_1[0]) != 0:
            P_Z_t_1[non_final_mask] = compute_P_Z_t_1(non_final_s_t_1)
        # Tgt support
        T_z = batch.r_t.unsqueeze(1) + (gamma * Z.unsqueeze(0)) * non_final_mask_float.unsqueeze(1)
        T_z = torch.clamp(T_z, min=V_min, max=V_max)
        # [m, num_atoms]
        b = (T_z - V_min) / dz
        # [m, num_atoms]
        l, u = b.floor(), b.ceil()
        # Handle case when b is int/at bin
        l -= (l == u).float()
        lt0 = (l < 0).float()
        l += lt0 
        u += lt0

        # [m, num_atoms]
        l_, u_ = l.long(), u.long()
        # [m, num_atoms]
        m_l = P_Z_t_1 * (u - b)
        m_u = P_Z_t_1 * (b - l)

        brange = range(m)
        pmf = torch.zeros((m, num_atoms), device=device)
        for i in range(num_atoms):
            pmf[brange, l_[brange, i]] += m_l[brange, i]
            pmf[brange, u_[brange, i]] += m_u[brange, i]
        # log can lead to nan()
        # kl_divs = -(pmf * P_Z_t.log()).sum(-1)
        # [m]
        kl_divs = -(pmf * log_P_Z_t).sum(-1)
        #if torch.isnan(kl_divs).any():
        #    ipdb.set_trace()
        return kl_divs
    return compute_kl_div


def create_optimize_f(
        t_config, n_steps, double_dqn, Z_support,
        dz, num_atoms, V_min, V_max
        ):
    dist_q_net, dist_tgt_net = t_config.models["q_net"], t_config.models["tgt_net"]
    m = t_config.batch_size
    batch2tensor_f = dqn_learner.create_batch2tensor_f(m, t_config.device)

    compute_P_Z_t_1_f = create_compute_P_Z_t_1_f(
            dist_q_net, dist_tgt_net, Z_support, num_atoms
            )

    compute_kl_div_f = create_compute_kl_div_f(
            compute_P_Z_t_1_f, dist_q_net, m, n_steps,
            t_config.gamma, 
            num_atoms, V_min, V_max, Z_support, dz,
            t_config.device
            )

    def optimize_f(batch, IS_weights):
        batch, non_final_mask = batch2tensor_f(batch)
        KL_divs = compute_kl_div_f(batch, non_final_mask)
        if IS_weights is not None:
            IS_weights = torch.tensor(IS_weights, dtype=torch.float32, device=t_config.device)
            loss = KL_divs * IS_weights
        else:
            loss = KL_divs
        t_config.optimizer.zero_grad()
        loss.mean().backward()
        for param in dist_q_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-t_config.grad_clip, t_config.grad_clip)
        t_config.optimizer.step()
        return KL_divs.detach().cpu().numpy()
    return optimize_f


def create_sample_optimize_f(
        t_config, n_steps, use_ddqn, Z_support, dz,
        num_atoms, V_min, V_max, replay_buffer, tracker
        ):
    optimize_f = create_optimize_f(
            t_config, n_steps, use_ddqn,
            Z_support, dz, num_atoms, V_min, V_max
            )
    track_f = dqn_learner.create_track_f(t_config.models["q_net"], tracker, "cross-entropy")
    if replay_buffer.sample_method == "td_err_prioritized":
        sample_optimize_f = dqn_learner.wrap_prio_sample(
                optimize_f, replay_buffer, t_config.batch_size, track_f
                )
    elif replay_buffer.sample_method == "uniform":
        sample_optimize_f = dqn_learner.wrap_uniform_sample(
                optimize_f, replay_buffer, t_config.batch_size, track_f 
                )
    return sample_optimize_f


