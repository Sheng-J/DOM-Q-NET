import multiprocessing as mp
from queue import Queue


REPLAY_SAMPLE = 0
REPLAY_EXIT = 1
REPLAY_UPDATE = 2


class ReplayProcess(mp.Process):
    def __init__(
            self, actor_queues, learner_conn,
            replay_buffer, m, max_push_per_sample
            ):
        super(ReplayProcess, self).__init__()
        self._actor_queues = actor_queues
        self._learner_conn = learner_conn
        self._replay_buffer = replay_buffer
        self._m = m
        self._n_push = max_push_per_sample

    def run(self):
        replay = self._replay_buffer
        while True:
            i = 0
            push_finish = False
            while not push_finish:
                push_finish = True
                for q in self._actor_queues:
                    try:
                        transitions = q.get()
                        for trans, prio in transitions:
                            replay.push_with_prio(trans, prio)
                            i += 1
                    except Queue.Empty:
                        continue
                    else:
                        raise ValueError("Somethingwrong...")
                    i += 1
                    # do next loop check if one q available for push
                    push_finish = False
                    if i >= self._n_push:
                        push_finish = True
                        break

            if self._learner_conn.poll() and len(replay_buffer) >= self._m:
                op, data = self._learner_conn.recv()
                if op == REPLAY_SAMPLE and replay.sample_method == "td_err_prioritized":
                    m, t_anneal = data
                    self._learner_conn.send(
                        replay.sample(m, t_anneal, t_display)
                        )
                    op, (idxs, prios) = self._learner_conn.recv()
                    if op == REPLAY_UPDATE:
                        replay.update_priorities(idxs, prios)
                    elif op == REPLAY_EXIT:
                        break
                    else:
                        raise ValueError("Only call UPDATE OR EXIT after sample")
                elif op == REPLAY_SAMPLE and replay.sample_method == "uniform":
                    self._learner_conn.send(
                        replay.sample(self._m)
                        )
                elif op == REPLAY_EXIT:
                    break
                else:
                    raise ValueError("Not supported op")
            

