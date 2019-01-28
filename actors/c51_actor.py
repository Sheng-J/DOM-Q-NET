import torch


def a_from_P_Z(P_Z_all_a, Z, mask=None):
    """
    -- output from qnet --
    -- dim1 = probs for the support Z_t --
    P_Z_all_a [m, num_a, num_atoms]

    -- fixed support Z --
    Z [num_atoms]

    -- Mask for certain unavailable actions --
    mask [m, num_a]
    """
    # [m, num_a] = [m, num_a, num_atoms] * [num_atoms]
    Q = torch.matmul(P_Z_all_a, Z)  
    if mask is not None:
        inf_mask = mask.clone()
        inf_mask[mask == 0] = float('inf')
        inf_mask[mask == 1] = 0
        Q -= inf_mask
    a = torch.max(Q, dim=1)[1]  # [m]
    return a  


def build_Z_support(V_min, V_max, num_atoms, device):
    dz = (V_max - V_min) / float((num_atoms - 1))
    Z_support = torch.arange(V_min, V_max + dz/2, dz, device=device)
    return Z_support, dz


def create_greedy_f(q_net, Z_support):
    def dist_q_policy(s_t, t, raw_s_t):
        """
        Expects dist_q_net forward -> (P_Z, mask)
        """
        with torch.no_grad():
            # [m, a_dim, num_atoms]
            P_Z, mask = q_net(s_t)
            a_t = a_from_P_Z(P_Z, Z_support, mask).view(1, 1)
            return a_t, None, P_Z
    return dist_q_policy








