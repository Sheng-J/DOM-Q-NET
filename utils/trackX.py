from tensorboardX import SummaryWriter
from utils.track import Tracker, create_mat_figure
import os
import ipdb


class TrackerX:
    def __init__(self, export_dir, entry_name, tracker):
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        self._tracker = tracker
        self._writer = SummaryWriter(export_dir)
        self._entry_name = entry_name

    @property
    def writer(self):
        return self._writer

    def fork(self, subdir_name):
        subdir_path = os.path.join(self._export_dir, subdir_name)
        tracker = self._tracker.fork(subdir_name)
        return TrackerX(subdir_path, self._entry_name+"_"+subdir_name, tracker)

    def tracks(self, keys, x, ys):
        for key, y in zip(keys, ys):
            self.track(key, x, y)

    def track(self, key, x, y):
        y = self._tracker.track(key, x, y)
        self._writer.add_scalars(key, {self._entry_name: y}, x)

    def track_t(self, key, x, y):
        self._tracker.track_t(key, x, y)
        self._writer.add_scalars(key, {self._entry_name: y}, x)

    def add_E(self, E, x, labels, tag='E'):
        self._writer.add_embedding(E, metadata=labels, global_step=x, tag=tag)

    def add_net(self, net, input_to_model):
        self._writer.add_graph(net, input_to_model)

    def add_fig(self, tag, figure, x):
        self._writer.add_figure(tag, figure, global_step=x)

    def add_img(self, tag, img_tensor, x):
        self._writer.add_image(tag, img_tensor, global_step=x)

    @property
    def track_T_dict(self):
        return self._tracker.track_T_dict


