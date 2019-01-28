"env": str - single task env
       list of strings - multitask envs, need to set multi_env true

"io_T": int - print Period
"print_last_n_avg" int - n for moving average
"T_replay_track" int - period for buffer stats 
"export_T" int - period for tracker dat format export of results



Optionals(default will be assigned in the program if not given):
"buffer_device": "gpu/cpu" - device on which processed transitions are stored
"batch_device": "gpu/cpu" - device on which backprop is performed
"eps_diff_print": if using eps greedy, print when eps decreased by this val
"multi_env": True/False - if performing multitasking

"save_ckpt": "True/False" default:False - whether to save ckpt or not
"T_ckpt": int - period(num of frames) for saving checkpoint







