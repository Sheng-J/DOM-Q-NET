import collections


class OnPiTrainConfig(collections.namedtuple(
    "TrainConfig",
    ("models", "optimizer", "gamma", "init_epi", "total_epi", "device")
    )):
    pass


class TrainConfig(collections.namedtuple(
    "TrainConfig",
    ("batch_size", "models", "optimizer", "grad_clip", "gamma",
     "init_step", "total_steps", "init_epi",
     "buffer_device", "batch_device", "max_step_per_epi", "save_f")
    )):
    def print_info(self):
        print("Batch size: %d\nGrad clip: %.1f\ngamma: %.2f"%(self.batch_size, self.grad_clip, self.gamma))
        print("max_step_per_epi=%d"%self.max_step_per_epi)


class EpsilonGreedyConfig(collections.namedtuple(
    "EpsilonGreedyConfig",
    ("eps_schedule_f", "action_space_f", "eps_print_diff", "t_exploration")
    )):
    pass


# Supervised configs
class DataConfig(collections.namedtuple(
    "DataConfig",
    ()
    )):
    pass


class DataTrainConfig(collections.namedtuple(
    "TrainConfig",
    ("optimizer", "criterion", "device", "save_f",
    "train_d_loader", "train_d_size", "test_d_loader", "test_d_size")
    )):
    pass


