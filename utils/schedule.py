import math


def create_constant_schedule(offset_val, offset_t, y):
    print("Schedule starts at %d " % offset_t)
    def schedule_f(t_anneal):
        if t_anneal < offset_t:
            return offset_val
        t_anneal -= offset_t
        return y
    return schedule_f


def create_linear_schedule(schedule_steps, y_0, y_T, offset_val=None, offset_t=0):
    print("Schedule starts at %d " % offset_t)
    def schedule_f(t_anneal):
        if t_anneal < offset_t:
            return offset_val
        t_anneal -= offset_t
        fraction = min(float(t_anneal) / schedule_steps, 1.0)
        return y_0 - fraction * (y_0 - y_T)
    return schedule_f


def create_expo_schedule(offset_val, offset_t, schedule_steps, y_0, y_T):
    print("Schedule starts at %d " % offset_t)
    def schedule_f(t_anneal):
        if t_anneal < offset_t:
            return offset_val
        t_anneal -= offset_t
        fraction = math.exp(-1. * t_anneal / schedule_steps)
        return y_T + fraction * (y_0 - y_T)
    return schedule_f





