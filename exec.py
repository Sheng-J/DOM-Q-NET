import os
import time
import shutil
import ipdb
import json
import argparse
import collections
# import entries.ggnn_dom_net.entry as ggnn_dom_net_entry
#import entries.q_eval.entry as q_eval_entry
#import entries.playground.entry as playground_entry
import entries.official.entry as official_entry
import entries.demo.entry as demo_entry


ENTRY_LOOKUP = {
        "official": official_entry.main,
        "demo": demo_entry.main
        }


def get_path(default_path, path):
    if os.path.basename(path) == path:
        return os.path.join(default_path, path)
    else:
        return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('entry', type=str)
    parser.add_argument('res_name', type=str)
    parser.add_argument('settings', type=str)
    parser.add_argument('--hparams', type=str, nargs='+')
    parser.add_argument('--paths', type=str, nargs='+')
    parser.add_argument('--prints', type=str, nargs='+')
    parser.add_argument('--reset', default=False, action='store_true')
    args = parser.parse_args()
    assert args.entry in ENTRY_LOOKUP
    main_f = ENTRY_LOOKUP[args.entry]

    res_dir = os.path.join("entries", args.entry, "results", args.res_name)
    if os.path.exists(res_dir) and args.reset:
        shutil.rmtree(res_dir)
        print("%s deleted" % res_dir)
    # ipdb.set_trace()
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    p = get_path(os.path.join("entries", args.entry, "settings"), args.settings)
    with open(p) as f:
        settings = json.load(f)

    hparams_list = []
    if args.hparams is not None:
        ps = [get_path(os.path.join("entries", args.entry, "hparams"), h_name)
              for h_name in args.hparams]
        for p in ps:
            with open(p) as f:
                hparams_list.append(json.load(f))

    paths_list = []
    if args.paths is not None:
        ps = [get_path(os.path.join("entries", args.entry, "paths"), p_name)
              for p_name in args.paths]
        for p in ps:
            ext = os.path.splitext(p)[1]
            if ext == ".json":
                with open(p) as f:
                    paths_list.append(json.load(f))
            else:
                paths_list.append(p)

    prints_dict = {"res_dir": res_dir}
    if args.prints is not None:
        ps = [get_path(os.path.join("entries", args.entry, "prints"), p_name)
              for p_name in args.prints]
        for p in ps:
            with open(p) as f:
                for key, val in json.load(f).items():
                    prints_dict[key] = val


    main_f(res_dir, settings, hparams_list, paths_list, prints_dict)


