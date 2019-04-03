# DOM-Q-NET: Grounded RL on Structured Language
> "DOM-Q-NET: Grounded RL on Structured Language" _International Conference on Learning Representations_ (2019). Sheng Jia, Jamie Kiros, Jimmy Ba. 
> [[arxiv]](https://arxiv.org/abs/1902.07257) [[openreview]](https://openreview.net/forum?id=HJgd1nAqFX) <br />

# Requirement
Need to download selenium & install chrome driver for selenium..

# Installation
1. Clone this repo
2. Download MiniWoB++ environment from the original repo https://github.com/stanfordnlp/miniwob-plusplus  <br />
and copy miniwob-plusplus/html folder to miniwob/html in this repo <br />
3. In fact, this html folder could be stored anywhere, but remember to perform one of the following actions: <br />
> * Set environment variable `"WOB_PATH"` to <br />
`"your-path-to-miniwob-plusplus"/html/miniwob` <br />
> * Directly modify the `base_url` on line 33 of instance.py to  <br />
"your-path-to-miniwob-plusplus"/html/miniwob <br />
In my case, `base_url='file:///h/sheng/DOM-Q-NET/miniwob/html/miniwob/'` <br />
# Run experiment
Experiment launch files are stored under `runs`
For example,
```
cd runs/hard2medium9tasks/
sh run1.sh
```
will launch a 11 multi-task (`social-media` `search-engine` `login-user` `enter-password` `click-checkboxes` `click-option` `enter-dynamic-text` `enter-text` `email-inbox-delete` `click-tab-2` `navigation-tree`) experiment.

# Acknowledgement
Credit to Dopamine for the implementation of prioritized replay used in dstructs/dopamine_segtree.py   <br />
