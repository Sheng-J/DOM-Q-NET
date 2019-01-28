import math
import collections
import random
import numpy as np
import ipdb
from utils import schedule
#from dstructs import segtree
#from dstructs import segment_tree


class Transition(collections.namedtuple(
    "Transition",
    ('s_t', 'a_t', 'r_t', 's_t_1', 'buffer_idx')
    )):
    pass


class ReplayBuffer(object):
    def __init__(self, capacity, trans_mode="Transition"):
        print("Buffer Capacity=%d" % capacity)
        self._capacity = capacity
        self._memory = []
        self._stats = []
        self._next_idx = 0
        self._num_pos_trans = 0
        self._num_sample_called = 0
        # buffer stats 
        self._num_pos_trans_per_batch = 0
        self._num_neg_trans_per_batch = 0
        self._num_unique_pos_trans_per_batch = 0
        self._num_unique_neg_trans_per_batch = 0
        self._num_sample_since_last_get_stats = 0

    def push(self, *args):
        trans_args = args[:4]
        raw_r_t = trans_args[2].item()
        if len(self._memory) < self._capacity:
            self._memory.append(None)
            self._stats.append(None)
            if len(self._memory) == self._capacity:
                print("Replay Buffer memory MAX %d" % self._capacity)
        saved_idx = self._next_idx
        self._stats_update_push(raw_r_t, saved_idx)
        self._memory[saved_idx] = Transition(*trans_args[:4], saved_idx)
        # Move circular cursor
        self._next_idx = (self._next_idx + 1) % self._capacity
        return saved_idx

    def _stats_update_push(self, raw_r_t, saved_idx):
        was_pos_trans = ((self._stats[saved_idx] is not None) and self._stats[saved_idx][0])
        is_pos_trans = (raw_r_t > 0)
        if was_pos_trans:
            self._num_pos_trans -= 1
        if is_pos_trans:
            self._num_pos_trans += 1
        # [pos?, num_times_sampled, stored_at, last_err]
        self._stats[saved_idx] = [is_pos_trans, 0, self._num_sample_called]

    def sample(self, m):
        sampled_transitions = random.sample(self._memory, m)
        self._stats_update_sample(sampled_transitions)
        batch = Transition(*zip(*sampled_transitions))
        return batch

    def _stats_update_sample(self, sampled_transitions):
        self._num_sample_called += 1
        pos_buffer_idxs = []
        neg_buffer_idxs = []
        for trans in sampled_transitions:
            buffer_idx = trans.buffer_idx
            self._stats[buffer_idx][1] += 1
            if self._stats[buffer_idx][0]:
                self._num_pos_trans_per_batch += 1
                pos_buffer_idxs.append(buffer_idx)
            else:
                self._num_neg_trans_per_batch += 1
                neg_buffer_idxs.append(buffer_idx)
        self._num_unique_pos_trans_per_batch += len(set(pos_buffer_idxs))
        self._num_unique_neg_trans_per_batch += len(set(neg_buffer_idxs))
        self._num_sample_since_last_get_stats += 1

    def get_stats(self):
        """
        Get stats info for the transitions in current buffer
        (old buffers kept stats but were thrown away)
        """
        avg_times_each_trans_in_curr_buffer_sampled = 0
        avg_times_each_pos_trans_in_curr_buffer_sampled = 0
        avg_times_each_neg_trans_in_curr_buffer_sampled = 0
        avg_times_in_buffer = 0
        for stat in self._stats:
            avg_times_each_trans_in_curr_buffer_sampled += stat[1]
            if stat[0]:
                avg_times_each_pos_trans_in_curr_buffer_sampled += stat[1]
            else:
                avg_times_each_neg_trans_in_curr_buffer_sampled += stat[1]
            avg_times_in_buffer += (self._num_sample_called - stat[2])
        # ipdb.set_trace()
        buffer_size = float(len(self._stats))
        num_pos_trans = float(self._num_pos_trans)
        num_neg_trans = (buffer_size - num_pos_trans)
        avg_times_each_trans_in_curr_buffer_sampled /= buffer_size
        if num_pos_trans > 0:
            avg_times_each_pos_trans_in_curr_buffer_sampled /= num_pos_trans
        if num_neg_trans > 0:
            avg_times_each_neg_trans_in_curr_buffer_sampled /= num_neg_trans
        avg_times_in_buffer /= buffer_size

        num_sample_since_last = float(self._num_sample_since_last_get_stats)
        avg_num_pos_trans_per_batch = self._num_pos_trans_per_batch / num_sample_since_last
        avg_num_neg_trans_per_batch = self._num_neg_trans_per_batch / num_sample_since_last
        avg_num_unique_pos_trans_per_batch = self._num_unique_pos_trans_per_batch /num_sample_since_last
        avg_num_unique_neg_trans_per_batch = self._num_unique_neg_trans_per_batch /num_sample_since_last
        self._num_sample_since_last_get_stats = 0
        self._num_pos_trans_per_batch = 0
        self._num_neg_trans_per_batch = 0
        self._num_unique_pos_trans_per_batch = 0
        self._num_unique_neg_trans_per_batch = 0
        return  avg_times_in_buffer, avg_times_each_trans_in_curr_buffer_sampled, \
                avg_times_each_pos_trans_in_curr_buffer_sampled, \
                avg_times_each_neg_trans_in_curr_buffer_sampled, \
                num_pos_trans, num_neg_trans, num_pos_trans/buffer_size, \
                avg_num_pos_trans_per_batch, avg_num_neg_trans_per_batch, \
                avg_num_unique_pos_trans_per_batch, avg_num_unique_neg_trans_per_batch

    def __len__(self):
        return len(self._memory)

    def get_status(self):
        return self._memory

    @property
    def capacity(self):
        return self._capacity

    def __getitem__(self, idx):
        return self._memory[idx]

    @property
    def sample_method(self):
        return "uniform"


#class PrioritizedReplayBuffer(ReplayBuffer):
#    @classmethod
#    def create(
#            cls, capacity, alpha, beta0, beta_anneal_T, prioritized_eps,
#            trans_mode="Transition", tracker=None
#            ):
#        schedule_f = schedule.create_linear_schedule(
#                beta_anneal_T, y_0=beta0, y_T=1.0
#                )
#        replay_buffer = cls(capacity, alpha, schedule_f, prioritized_eps, trans_mode, tracker)
#        return replay_buffer
#
#    def __init__(self, capacity, alpha, beta_scheduler, prioritized_eps, trans_mode="Transition", tracker=None):
#        """
#        Extra parameters:
#            [1]: alpha for how much prioritization used
#            [2]: beta for compensate the bias due to non-uniform sampling
#
#        -alpha=0 no prioritization,
#        -alpha=1 full prioritization
#        """
#        it_capacity = 1
#        while it_capacity < capacity:
#            it_capacity *= 2
#
#        super(PrioritizedReplayBuffer, self).__init__(it_capacity, trans_mode)
#        assert alpha >= 0
#        self._alpha = alpha
#        #self._it_sum = segtree.SumSegmentTree(it_capacity)
#        #self._it_min = segtree.MinSegmentTree(it_capacity)
#        self._it_sum = segment_tree.SumSegmentTree(it_capacity)
#        self._it_min = segment_tree.MinSegmentTree(it_capacity)
#        self._max_priority = 1.0
#
#        self._beta_scheduler = beta_scheduler
#        self._prioritized_eps = prioritized_eps
#
#        self._tracker = tracker
#
#    def set_tracker(self, tracker):
#        self._tracker = tracker
#
#    def push_with_prio(self, trans, prio):
#        idx = self._next_idx
#        prio = math.abs(prio) + self._prioritized_eps
#        super().push(*trans)
#        self._it_sum[idx] = prio ** self._alpha
#        self._it_min[idx] = prio ** self._alpha
#
#    def push(self, *args):
#        """
#        args: [s_t, a_t, r_t, s_t_1]
#        """
#        idx = self._next_idx
#        super().push(*args)
#        # MIGHT BE SLOW
#        # TODO THIS IS NOT CORRECT 
#        max_priority = max(self._it_sum._value[self._it_sum._capacity:])
#        if max_priority == 0.:
#            max_priority = self._max_priority
#        #self._it_sum[idx] = self._max_priority ** self._alpha
#        #self._it_min[idx] = self._max_priority ** self._alpha
#        self._it_sum[idx] = max_priority ** self._alpha
#        self._it_min[idx] = max_priority ** self._alpha
#
#    def _sample_proportional(self, m):
#        idxes = []
#        temp_prios = []
#        # TODO no repeat ????
#        for _ in range(m):
#            # mass = random.random() * self._it_sum.sum(0, len(self._memory)-1)
#            mass = random.random() * self._it_sum.sum(0, len(self._memory))
#            # TODO remove this debugging later
#            #if math.isnan(mass):
#            #    ipdb.set_trace()
#            idx = self._it_sum.find_prefixsum_idx(mass)
#            idxes.append(idx)
#            ### AVOID DUPL
#            # temp_prios.append(self._it_sum[idx]**(1./self._alpha)) #TEMP
#            # self.update_priorities([idx], [0.]) #TEMP
#        # self.update_priorities(idxes, temp_prios) #TEMP
#        ###
#        return idxes
#
#    def sample(self, m, t_anneal):
#        """
#        beta: To what degree to use importance weights
#        (0 - no corrections, 1 -full correction)
#
#        Returns (transition, importance weights, idxes)
#        """
#        beta = 0 # self._beta_scheduler(t_anneal)
#        # self._tracker.track("prioritized_beta", t_anneal, beta)
#        # self._tracker.track("prioritized_alpha", t_anneal, self._alpha)
#
#        idxes = self._sample_proportional(m)
#
#        ws = []
#        p_min = self._it_min.min() / self._it_sum.sum()
#        N = len(self)
#        w_max = (p_min * N) ** (-beta)
#
#        prios = []
#        for idx in idxes:
#            p_sample = self._it_sum[idx] / self._it_sum.sum()
#            prios.append(self._it_sum[idx])
#            w_i = (p_sample * N) ** (-beta)
#            ws.append(w_i / w_max)
#        # ws = np.array(ws)
#        ws = np.array(ws, dtype=np.float32)
#
#        # ipdb.set_trace()
#        sampled_transitions = [self._memory[idx] for idx in idxes]
#        batch = Transition(*zip(*sampled_transitions))
#        # ipdb.set_trace()
#        return batch, ws, idxes
#
#    def update_priorities(self, idxes, priorities):
#        """
#        idxes: 1D list of indices
#        td_errors: 1D list of priorities(td errors + abs)
#        """
#        assert len(idxes) == len(priorities)
#        for idx, priority in zip(idxes, priorities):
#            if priority < 0:
#                raise ValueError("priority %.3f is wrong" % priority)
#            if not (0 <= idx < len(self)):
#                raise ValueError("idx %d is wrong" % idx)
#            self._it_sum[idx] = priority ** self._alpha
#            self._it_min[idx] = priority ** self._alpha
#
#            self._max_priority = max(self._max_priority, priority)
#
#    def update_priorities_from_err(self, idxes, errs):
#        """
#        idxes: 1D list of indices
#        td_errors: 1D list of td errors
#        """
#        # SET POINT
#        priorities = np.abs(errs) + self._prioritized_eps
#        self.update_priorities(idxes, priorities)
#
#    @property
#    def sample_method(self):
#        return "td_err_prioritized"
#

def create_track_f(replay_buffer, T_replay_track):
    def replay_track_f(tracker, t):
        if t>0 and t % T_replay_track == 0:
            track_keys = [
                    "avg_times_in_buffer_per_tran",
                    "avg_times_sampled_per_tran", "avg_times_sampled_per_pos_tran",
                    "avg_times_sampled_per_neg_tran", "num_pos_trans", "num_neg_trans",
                    "frac_pos_trans", "avg_num_pos_trans_per_batch", "avg_num_neg_trans_per_batch",
                    "avg_num_unique_pos_trans_per_batch", 
                    "avg_num_unique_neg_trans_per_batch"
                    ]
            # "max_q", "min_q", "avg_q"
            track_vals = replay_buffer.get_stats()
            tracker.tracks(track_keys, t, track_vals)
    def prio_replay_track_f(tracker, t):
        if t>0 and t % T_replay_track == 0:
            track_keys = [
                    "avg_times_in_buffer_per_tran",
                    "avg_times_sampled_per_tran", "avg_times_sampled_per_pos_tran",
                    "avg_times_sampled_per_neg_tran", "num_pos_trans", "num_neg_trans",
                    "frac_pos_trans", "avg_pos_trans_err", "avg_neg_trans_err",
                    "avg_num_pos_trans_per_batch", "avg_num_neg_trans_per_batch",
                    "avg_num_unique_pos_trans_per_batch", 
                    "avg_num_unique_neg_trans_per_batch", 
                    ]
            # "max_q", "min_q", "avg_q"
            track_vals = replay_buffer.get_stats()
            tracker.tracks(track_keys, t, track_vals)
    if replay_buffer.sample_method == "td_err_prioritized":
        return prio_replay_track_f
    else:
        return replay_track_f











