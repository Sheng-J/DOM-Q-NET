"opt_type" str - optimization type
"lr": float - learning rate,
"batch_size" int - batch size used for training when sampling from buffer,
"gamma": float - discount factor,
"T_tgt_update": int - DQN tgt network update period in number of frames, 
"T_update": int - DQN (gradient) update period in number of frames,
"max_step_per_epi": int - maximum num of steps before termination per epi,
                          terminated transition will have reward 0. since this 
                          is an off-policy learning, non episodic trans allowed
"T_exploration": int - number of random exploration steps before using policy,
"grad_clip": float - gradient clip,
"n_steps": int - number of steps for n-steps dqn, 
"T_total": - total number of frames for training,
"use_noisylayers": - use of noisy layers





