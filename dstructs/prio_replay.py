from dstructs import replay, dopamine_segtree
import numpy as np
import ipdb


class Buffer(replay.ReplayBuffer):
    def __init__(self, capacity, trans_mode="Transition", tracker=None):
        super(Buffer, self).__init__(capacity, trans_mode)
        self.sum_tree = dopamine_segtree.SumTree(capacity)
        self._tracker = tracker

    def push(self, *args):
        # stores prio=(TD_err+1e-10)**alpha
        idx = self._next_idx
        saved_idx = super().push(*args)
        # [idx=3, TD_err]
        err = np.power(self.sum_tree.max_recorded_priority, 1.25)
        self._stats[saved_idx].append(err)
        self.sum_tree.set(idx, self.sum_tree.max_recorded_priority)

    def sample(self, m):
        # Sample stratified indices. Some of them might be invalid.
        # list of int
        idxes = self.sum_tree.stratified_sample(m)

        # get transitions from idxes
        # ipdb.set_trace()
        sampled_transitions = [self._memory[idx] for idx in idxes]
        self._stats_update_sample(sampled_transitions)
        batch = replay.Transition(*zip(*sampled_transitions))

        # Compute IS weights
        # ndarray
        priorities = self.get_priority(idxes)
        IS_weights = 1.0 / np.power((priorities + 1e-10), 0.80)
        # ipdb.set_trace()
        IS_weights /= np.max(IS_weights)
        return batch, IS_weights, idxes

    def update_priorities(self, idxes, td_errs):
        # ipdb.set_trace()
        priorities = np.power((td_errs + 1e-10), 0.80)
        # priorities = np.sqrt(td_errs + 1e-10)
        for i, idx in enumerate(idxes):
            self.sum_tree.set(idx, priorities[i])
            self._stats[idx][3] = td_errs[i]

    def get_priority(self, indices):
        m = len(indices)
        priority_batch = np.empty((m), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self.sum_tree.get(memory_index)
        # list of float
        return priority_batch

    def get_stats(self):
        """
        Get stats info for the transitions in current buffer
        (old buffers kept stats but were thrown away)
        """
        avg_times_each_trans_in_curr_buffer_sampled = 0
        avg_times_each_pos_trans_in_curr_buffer_sampled = 0
        avg_times_each_neg_trans_in_curr_buffer_sampled = 0
        avg_pos_trans_err = 0
        avg_neg_trans_err = 0
        avg_times_in_buffer = 0
        for stat in self._stats:
            avg_times_each_trans_in_curr_buffer_sampled += stat[1]
            if stat[0]:
                avg_times_each_pos_trans_in_curr_buffer_sampled += stat[1]
                avg_pos_trans_err += stat[3]
            else:
                avg_times_each_neg_trans_in_curr_buffer_sampled += stat[1]
                avg_neg_trans_err += stat[3]
            avg_times_in_buffer += (self._num_sample_called - stat[2])
        # ipdb.set_trace()
        buffer_size = float(len(self._stats))
        num_pos_trans = float(self._num_pos_trans)
        num_neg_trans = (buffer_size - num_pos_trans)
        avg_times_each_trans_in_curr_buffer_sampled /= buffer_size
        if num_pos_trans > 0:
            avg_times_each_pos_trans_in_curr_buffer_sampled /= num_pos_trans
            avg_pos_trans_err /= num_pos_trans
        if num_neg_trans > 0:
            avg_times_each_neg_trans_in_curr_buffer_sampled /= num_neg_trans
            avg_neg_trans_err /= num_neg_trans
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
                avg_pos_trans_err, avg_neg_trans_err, \
                avg_num_pos_trans_per_batch, avg_num_neg_trans_per_batch, \
                avg_num_unique_pos_trans_per_batch, avg_num_unique_neg_trans_per_batch

    @property
    def sample_method(self):
        return "td_err_prioritized"

    def set_tracker(self, tracker):
        self._tracker = tracker
