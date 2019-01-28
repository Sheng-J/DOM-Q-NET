cd ../..
python3 exec.py official test_align_raw \
    settings/login_user.json \
    --paths paths/loginuser.json \
    --hparams hparams/nn/1.json \
    hparams/qlearn/ddqn_nsteps_noisy_layers0_7.json \ 
    hparams/replay_buffer/prioritized0.json \
    hparams/graph/3.json \
    hparams/dom/goal_attn1.json \ 
    --reset
cd - 
