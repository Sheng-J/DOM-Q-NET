from pprint import pprint
import ipdb


class Qdebugger:
    def __init__(self, q_net, memory, debug_interval, tgt_net=None):
        self._interval = debug_interval
        self._q_net = q_net
        self._memory = memory
        self._records = {"q":{}, "ep":{}}
        self._records_lookup = [None for _ in range(self._memory.capacity)]
        self._last_recorded_step = None

    def record_step(self, mem_idx, q_doms, eps, step, R_t):
        prev = self._records_lookup[mem_idx]
        if prev:
            try:
                del(self._records[prev[0]][prev[1]][prev[2]][prev[3]])
            except:
                ipdb.set_trace()
                pass

        if q_doms is not None:
            record_type = "q"
        else:
            record_type = "ep"
        if R_t not in self._records[record_type]:
            self._records[record_type][R_t] = {}
        if eps not in self._records[record_type][R_t]:
            self._records[record_type][R_t][eps] = {}
        self._records[record_type][R_t][eps][step] = (q_doms, mem_idx)
        self._records_lookup[mem_idx] = (record_type, R_t, eps, step)
        self._last_recorded_step = (eps, step)

    def get_records_for_reward(self, reward):
        q_records = self._records["q"].get(reward)
        ep_records = self._records["ep"].get(reward)
        return q_records, ep_records

    def debug_full_status(self, eps):
        if eps % self._interval == 0:
            pprint(self._memory.get_status())
            pprint(self._q_net.get_status())
            ipdb.set_trace()

    def trace_q_non_pos_reward(self):
        rewards = self._records["q"].keys()
        non_pos_rewards = []
        for r in rewards:
            if r <= 0:
                non_pos_rewards.append(r)
        if len(non_pos_rewards) > 0:
            lookups = {reward: self._records["q"][reward] for reward in non_pos_rewards}
            print("Debugging non pos q:")
            ipdb.set_trace()
            print("Finished debugging")

    @property
    def last_recorded_step(self):
        return self._last_recorded_step



