import math
import torch
import torch.nn.functional as F
import numpy as np
import ipdb
from pprint import pprint
from dstructs import replay

# For intrinsic and extrinsic rewards, Two TD tgts.  Two updates for two separate
# Q heads to update.  So intrinsic rewards used here once.
# Need one  

# For forward loss,  
# When computing priorities, sum [1]TD error intrin[2]TD error ext[3]forwardloss  
#[1,2] for Qnet update.  [3] for predictor update.  Intrinsic reward used for Qnet update  
#[1,2] for            [3]

def create_batch2tensor_f(m, buffer_device_type, batch_device):
    def batch2tensor(batch):
        # batch.s_t [m(list), num_input_items_for_forward, dim_for_each_item*]
        num_items = len(batch.s_t[0])
        batch_s_t = [[] for _ in range(num_items)]
        non_final_batch_s_t_n = [[] for _ in range(num_items)]
        non_final_mask = []
        # batch.a_t [m, 1]
        a_t_tensors = torch.cat(batch.a_t, dim=0)
        # batch.r_t [m]
        r_t_n_tensors = torch.cat(batch.r_t)
        # ipdb.set_trace()
        # transpose trans, batch each
        # list in the form of [m, num_input_items, dim for each item]
        # to                  [num_input_items, m, dim]
        for batch_id in range(m):
            non_final_mask.append(batch.s_t_1[batch_id] is not None)
            if batch.s_t_1[batch_id] is not None:
                for item_id in range(num_items):
                    batch_s_t[item_id].append(batch.s_t[batch_id][item_id])
                    non_final_batch_s_t_n[item_id].append(batch.s_t_1[batch_id][item_id])
            else:
                for item_id in range(num_items):
                    batch_s_t[item_id].append(batch.s_t[batch_id][item_id])
        s_t_tensors = [torch.cat(batch_s_t[i], dim=0) for i in range(num_items)]
        # ipdb.set_trace()
        # [m]
        non_final_mask = torch.tensor(
            non_final_mask,
            device=batch_device, dtype=torch.uint8
            )
        # print(sum(non_final_mask))  <- BUG! produces 0 and not cat even tho
        # non_final_batch_s_t_n is not empty
        if non_final_mask.sum().item() != 0:
            # [num_input_items, m_not_final, dim]
            non_final_s_t_n_tensors = [torch.cat(non_final_batch_s_t_n[i], dim=0) for i in range(num_items)]
        else:
            # [num_input_items, 0]
            non_final_s_t_n_tensors = non_final_batch_s_t_n

        # s_t_tensors[num_items, m, d], a_t_tensors[m, 1], r_t_n_tensors [m],
        # non_final_tensors [num_items, m_non_final, d]
        if batch_device.type != buffer_device_type:
            s_t_tensors = [item.to(batch_device) for item in s_t_tensors]
            #a_t_tensors = [item.to(batch_device) for item in a_t_tensors]
            #r_t_n_tensors = [item.to(batch_device) for item in r_t_n_tensors]
            if non_final_mask.sum().item() != 0:
                # Only move and used later when non_final_s_t_n tensors not
                # empty
                non_final_s_t_n_tensors = [item.to(batch_device) for item in non_final_s_t_n_tensors]
            else:
                # [num_input_items, 0]
                non_final_s_t_n_tensors = non_final_batch_s_t_n
            #a_t_tensors = [item.to(batch_device) for
            #s_t_tensors = s_t_tensors.to(batch_device)
            a_t_tensors = a_t_tensors.to(batch_device)
            r_t_n_tensors = r_t_n_tensors.to(batch_device)
            #non_final_s_t_n_tensors = non_final_s_t_n_tensors.to(batch_device)
        batch_trans_tensors = replay.Transition(s_t_tensors, a_t_tensors, r_t_n_tensors, non_final_s_t_n_tensors, None)
        # ipdb.set_trace()
        return batch_trans_tensors, non_final_mask
    return batch2tensor


def create_compute_Q_t_1_f(q_net, tgt_net, use_ddqn):
    # Configure Q(S_t_n, a) computing function
    if use_ddqn:
        # compute_Q_t_1 = lambda S_t: tgt_net(S_t)[q_net(S_t).max(dim=1)[1]].detach()
        print("use ddqn")
        def compute_Q_t_1(s_t_1):
            q_ts_1_list = q_net(s_t_1)
            tgt_q_ts_1_list = tgt_net(s_t_1)
            selected_q_ts_1_list = []
            for q_t_1_, tgt_q_t_1_ in zip(q_ts_1_list, tgt_q_ts_1_list):
                a_t_1 = q_t_1_.max(dim=1, keepdim=True)[1]
                q_t_1 = tgt_q_t_1_.gather(dim=1, index=a_t_1).detach()
                selected_q_ts_1_list.append(q_t_1.squeeze(1))
            return selected_q_ts_1_list
    else:
        print("NOT use ddqn")
        #TODO NEED UPDATE
        compute_Q_t_1 = lambda s_t_1: tgt_net(s_t_1).max(dim=1)[0].detach()
    return compute_Q_t_1


def create_compute_td_err_f(compute_Q_t_1, q_net, m, n_steps, gamma, batch_device):
    def compute_td_err(batch, non_final_mask):
        # [m], [m], [m]
        a_ts = batch.a_t
        num_action_types = a_ts.size()[1]
        TD_errs_list = []
        s_t, non_final_s_t_1 = batch.s_t, batch.s_t_1
        if len(non_final_s_t_1[0]) != 0:
            # qvalues [m'],  [m'], [m']
            q_ts_1_list = compute_Q_t_1(non_final_s_t_1)
        # q_dom,     q_acttype,      q_text
        # [m, max_num_doms],  [m, 2], [m, max_num_goals]
        q_t_all_a_list = q_net(s_t)
        A_q_t_s = []
        TD_TGTS = []
        for i in range(num_action_types):
            a_t = a_ts[:, i].unsqueeze(1)
            # Q_t_all
            q_t_all_a = q_t_all_a_list[i]   # [m, s]
            q_t = q_t_all_a.gather(1, a_t).squeeze(1) # [m]
            # Y_Q_t
            q_t_1 = torch.zeros(m, device=batch_device) # [m]
            # If item has at least one non final s_t_1
            if len(batch.s_t_1[0]) != 0:
                q_t_1[non_final_mask] = q_ts_1_list[i]
            TD_tgt = batch.r_t + (gamma**n_steps) * q_t_1
            A_q_t_s.append(q_t)
            TD_TGTS.append(TD_tgt)
            # TD Error Update [m]
            #TD_errs = F.smooth_l1_loss(q_t, TD_tgt, reduce=False)
            # TD_errs_list.append(TD_errs)
        A = A_q_t_s[0] + A_q_t_s[1] + A_q_t_s[2]
        B = TD_TGTS[0] + TD_TGTS[1] + TD_TGTS[2]
        # ipdb.set_trace()
        TD_errs = F.smooth_l1_loss(A, B, reduce=False)
        TD_errs_list.append(TD_errs)
        # TD_errs_list.append(TD_errs)
        # TD_errs_list.append(TD_errs)
        return TD_errs_list
    return compute_td_err

def create_optimize_f(t_config, n_steps, use_ddqn):
    q_net, tgt_net = t_config.models["q_net"], t_config.models["tgt_net"]
    m = t_config.batch_size
    batch2tensor_f = create_batch2tensor_f(m, t_config.buffer_device.type, t_config.batch_device)

    compute_Q_t_1_f = create_compute_Q_t_1_f(q_net, tgt_net, use_ddqn)

    # TD error computing function
    compute_td_err_list_f = create_compute_td_err_f(
            compute_Q_t_1_f, q_net, m, n_steps,
            t_config.gamma, t_config.batch_device
            )

    def optimize_f(batch, IS_weights=None):
        batch, non_final_mask = batch2tensor_f(batch)
        TD_errs_list = compute_td_err_list_f(batch, non_final_mask)
        detached_TD_errs_list =  []
        t_config.optimizer.zero_grad()
        loss_sum = 0
        for TD_errs in TD_errs_list:
            if IS_weights is not None:
                IS_weights = torch.tensor(IS_weights, dtype=torch.float32, device=t_config.batch_device)
                # TODO IS_weights TD_errs same unique size?
                # ipdb.set_trace()
                # [m] = [m] * [m]
                loss = TD_errs * IS_weights
            else:
                loss = TD_errs
            loss_sum += loss.mean()
            # [m]
            detached_err = TD_errs.detach().cpu().numpy()
            detached_TD_errs_list.append(detached_err)
        loss_sum.backward()
        for param in q_net.parameters():
            # TODO None?
            if param.grad is not None:
                param.grad.data.clamp_(-t_config.grad_clip, t_config.grad_clip)
        t_config.optimizer.step()
        return np.sum(detached_TD_errs_list, 0)
    return optimize_f


def wrap_prio_sample(optimize_f, replay_buffer, m, track_f):
    def step(t):
        # batch_trans, IS_weights, idxes = replay_buffer.sample(m, t)
        batch_trans, IS_weights, idxes = replay_buffer.sample(m)
        errs = optimize_f(batch_trans, IS_weights)
        #for err in errs:
        #    if math.isnan(err):
        #        ipdb.set_trace()
        replay_buffer.update_priorities(idxes, errs)
        track_f(t, errs)
    return step


def wrap_uniform_sample(optimize_f, replay_buffer, m, track_f):
    def step(t):
        batch_trans = replay_buffer.sample(m)
        errs = optimize_f(batch_trans)
        track_f(t, errs)
    return step


def create_track_f(q_net, tracker, label="TD_err"):
    T_track = tracker.track_T_dict[label]
    def track_f(t, td_errs):
        if t % T_track == 0:
            # ipdb.set_trace()
            tracker.tracks([label], t, [td_errs.mean()])
    return track_f


def create_sample_optimize_f(t_config, n_steps, use_ddqn, replay_buffer, tracker=None):
    optimize_f = create_optimize_f(t_config, n_steps, use_ddqn)
    track_f = create_track_f(t_config.models["q_net"], tracker)
    if replay_buffer.sample_method == "td_err_prioritized":
        sample_optimize_f = wrap_prio_sample(
                optimize_f, replay_buffer, t_config.batch_size, track_f
                )
    elif replay_buffer.sample_method == "uniform":
        sample_optimize_f = wrap_uniform_sample(
                optimize_f, replay_buffer, t_config.batch_size, track_f 
                )
    return sample_optimize_f


