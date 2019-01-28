cd ../..
python3 exec.py "entry point name" "run name"  \
    settings/"settings json file"  \
    --paths paths/"path files"  \
    --hparams hparams/nn/"nn hparams json file"  \
    hparams/qlearn/"qlearn hparams json file"   \
    hparams/replay_buffer/"replay buffer json file"  \
    hparams/graph/"graph json file"   \
    hparams/dom/"dom json file"   \
    --reset
cd -


