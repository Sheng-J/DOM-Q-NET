cd ../..
python3.6 exec.py official hard2medium9_attn_cat_ckpt1  \
    settings/hard2medium9tasks_cpu_saveckpt.json  \
    --paths paths/hard2medium9tasks.json  \
    --hparams hparams/nn/multitask_vocab600.json  \
    hparams/qlearn/ddqn_nsteps_multitask.json   \
    hparams/replay_buffer/multitask_prioritized100000.json  \
    hparams/graph/7.json   \
    hparams/dom/goal_attn_cat_large_V.json   \
    --reset
cd -


