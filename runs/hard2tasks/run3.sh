cd ../..
python3 exec.py official cat_JAN_26th_1159AM  \
    settings/hard2tasks_cpu.json  \
    --paths paths/hard2tasks.json  \
    --hparams hparams/nn/multitask_large_vocab.json  \
    hparams/qlearn/ddqn_nsteps_multitask.json   \
    hparams/replay_buffer/multitask_prioritized100000.json  \
    hparams/graph/7.json   \
    hparams/dom/goal_cat_large_V.json   \
    --reset
cd -
