import ipdb


class EnvWrap:
    def __init__(self, env):
        self._env = env
        self._epi_step = 0
        self._epi_reward = 0

    @property
    def epi_step(self):
        return self._epi_step

    @property
    def epi_reward(self):
        return self._epi_reward

    def reset(self):
        self._epi_step = 0
        self._epi_reward = 0
        items_tuple = self._env.reset()
        return items_tuple

    def step(self, a):
        items_tuple = self._env.step(a)
        self._epi_step += 1
        self._epi_reward += items_tuple[1]
        return items_tuple

