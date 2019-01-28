import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._std_init = std_init
        self._W_mu = nn.Parameter(torch.empty(out_dim, in_dim))
        self._W_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        self.register_buffer('_W_eps', torch.empty(out_dim, in_dim))
        self._b_mu = nn.Parameter(torch.empty(out_dim))
        self._b_sigma = nn.Parameter(torch.empty(out_dim))
        self.register_buffer('_b_eps', torch.empty(out_dim))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self._in_dim)
        self._W_mu.data.uniform_(-mu_range, mu_range)
        self._W_sigma.data.fill_(self._std_init / math.sqrt(self._in_dim))
        self._b_mu.data.uniform_(-mu_range, mu_range)
        self._b_sigma.data.fill_(self._std_init / math.sqrt(self._out_dim))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_p = self._scale_noise(self._in_dim)
        epsilon_q = self._scale_noise(self._out_dim)
        # q[out_dim, 1] * p[1, in_dim] = [out_dim, in_dim]
        self._W_eps.copy_(epsilon_q.ger(epsilon_p))
        # q[out_dim, 1]
        self._b_eps.copy_(epsilon_q)

    def forward(self, X):
        if self.training:
            Y = F.linear(
                    X,
                    self._W_mu + self._W_sigma * self._W_eps,
                    self._b_mu + self._b_sigma * self._b_eps
                    )
        else:
            Y = F.linear(
                    X,
                    self._W_mu,
                    self._b_mu
                    )
        return Y
        


