cd ../..
python3.6 exec.py official hard2tasks_attn_cat_JAN_24th_1020PM  \
    settings/hard2tasks_cpu.json  \
    --paths paths/hard2tasks.json  \
    --hparams hparams/nn/multitask_large_vocab.json  \
    hparams/qlearn/ddqn_nsteps_multitask.json   \
    hparams/replay_buffer/multitask_prioritized100000.json  \
    hparams/graph/7.json   \
    hparams/dom/goal_attn_cat_large_V.json   \
    --reset
cd -


