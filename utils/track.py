import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


class Tracker:
    def __init__(
            self,
            export_dir,
            io_T_dict,
            last_n_ave_dict,
            export_T,
            decimal_dict,
            track_T_dict,
            track_name=""):
        """
        records {key:[[x-time], [y-val]]}
        E.g. reward, step 'tracked' after each episode, but step is x
             batch_mean_td_errs, frac of pos_rewards, 'tracked' after each 
             frac of pos_rewards in batch
        """
        plt.ion()
        self.track_T_dict = track_T_dict
        self._export_dir = export_dir
        self._track_keys = io_T_dict.keys()
        # raw record
        self._records = {}
        # last n ave record if n_ave provided for key
        self._ave_records = {}
        # last t, for x recorded
        self._last_io_x = {}
        self._last_export_x = 0
        self._figure_ids = {}
        self._counters = {}
        for key in self._track_keys:
            self._records[key] = [[], []]
            self._ave_records[key] = [[], []]
            self._figure_ids[key] = len(self._records)
            self._last_io_x[key] = 0
        # Dicts for storing T, period
        self._io_T_dict = io_T_dict # (mandatory)
        self._last_n_ave_dict = last_n_ave_dict # (optional)
        self._export_T = export_T # (mandatory)
        # decimal for display (Mandatory for each)
        self._decimal_dict = decimal_dict
        self._track_name = track_name

    def fork(self, subdir_name):
        subdir_path = os.path.join(self._export_dir, subdir_name)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        return Tracker(
                subdir_path, self._io_T_dict, self._last_n_ave_dict,
                self._export_T, self._decimal_dict, self.track_T_dict)

    def set_export_dir(self, export_dir):
        self._export_dir = export_dir

    def tracks(self, keys, x, ys):
        for key, y in zip(keys, ys):
            self.track(key, x, y)

    def _track(self, key, x, y):
        """
        returned value is y if last_n_ave is None
        o.w. overriden
        """
        self._records[key][0].append(x)
        self._records[key][1].append(y)
        last_n = None
        if key in self._last_n_ave_dict:
            self._ave_records[key][0].append(x)
            items = (self._records[key][1])[-self._last_n_ave_dict[key]:]
            last_n = len(items)
            y = sum(items) / last_n
            self._ave_records[key][1].append(y)
        if (x - self._last_export_x) >= self._export_T:
            self.export_data()
            self._last_export_x = x
        return y, last_n

    def track(self, key, x, y):
        # If last_n_ave has the key, y overriden as last n ave
        y, last_n = self._track(key, x, y)
        if (x - self._last_io_x[key]) >= self._io_T_dict[key]:
            if last_n is not None:
                info = "%d_%s"%(last_n, key)
            else:
                info = key
            dec_str = str(self._decimal_dict[key])
            print(("%s key=%s\t\ty=%." + dec_str + "f\t\t@ x=%d")%(self._track_name, info, y, x))
            self._last_io_x[key] = x
        return y

    def count(self, key, num_inc=1):
        if key not in self._counters:
            self._counters[key] = 1
        self._counters[key] += num_inc

    def get_count(self, key):
        return self._counters.get(key)

    def export_data(self):
        assert self._export_dir is not None
        def export(records, data_type=""):
            for key, (X, Y) in records.items():
                with open(os.path.join(self._export_dir, "%s_%s.dat"%(key, data_type)), "w") as f:
                    for x, y in zip(X, Y):
                        f.write("%.2f %.2f\n"%(x, y))
        export(self._records)
        export(self._ave_records, "avg")
        #if self._last_n_ave is not None:
        #    export(self._ave_records, "%d_ave"%self._last_n_ave)

    def plot(self, *keys):
        for key in keys:
            plt.figure(self._figure_ids[key])
            plt.clf()
            plt.plot(self._records[key][0], self._records[key][1])
            plt.title(key)
            plt.pause(0.001)

    def plot_all(self, pause=False):
        self.plot(*self._records.keys())
        if pause:
            plt.show()

    def plot_last_n_ave(self, key, n, x_label, y_label, title):
        i = n
        xs, ys = self._records[key]
        vals = []
        while i<len(xs):
            ave = sum(ys[i-n:i])/n
            vals.append(ave)
            i+=1
        
        plt.figure(self._figure_ids[key])
        plt.clf()
        plt.plot(self._records[key][0][n:], vals)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(title+".jpg")


    def __del__(self):
        plt.ioff()



def plot_last_n_ave(trackers, key, n, res_names, x_label, y_label, title):
    for tracker, res_name in zip(trackers, res_names):
        i = n
        xs, ys = tracker._records[key]
        vals = []
        while i<len(xs):
            ave = sum(ys[i-n:i])/n
            vals.append(ave)
            i+=1
        plt.plot(tracker._records[key][0][n:], vals, label=res_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, color='k')
    plt.savefig(title+".jpg")


def plot_last_n_ave2(data_paths, n, N, res_names, x_label, y_label, title):
    for data_path, res_name in zip(data_paths, res_names):
        xs, ys = [], []
        with open(data_path) as f:
            for line in f:
                x_y = line.split()
                xs.append(float(x_y[0]))
                ys.append(float(x_y[1]))
            ave_ys = []
            for i in range(N-1, len(ys)):
                ave_ys.append(sum(ys[i-(N-1):i+1])/N)
            plt.plot(xs[N-1:], ave_ys, label=res_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend()
    plt.grid(True, color='k')
    plt.savefig(title+".jpg")


def plot_last_n_ave3(data_paths, n, N, res_names, x_label, y_label, title):
    for data_path, res_name in zip(data_paths, res_names):
        xs, ys = [], []
        with open(data_path) as f:
            for line in f:
                x_y = line.split()
                xs.append(float(x_y[0]))
                ys.append(float(x_y[1]))
            ave_ys = []
            for i in range(len(ys)):
                items = ys[max(0, i-(N-1)):i+1]
                ave_ys.append(sum(items)/len(items))
            plt.plot(xs[10:n], ave_ys[10:n], label=res_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(-1, 1)
    plt.title(title)
    plt.legend()
    plt.grid(True, color='k')
    plt.savefig(title+".jpg")



def create_mat_figure(tensor, labels):
    plt.cla()
    plt.clf()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(tensor, aspect='equal')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.canvas.draw()
    return fig


