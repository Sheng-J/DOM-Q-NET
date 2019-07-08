# DOM-Q-NET: Grounded RL on Structured Language
> "DOM-Q-NET: Grounded RL on Structured Language" _International Conference on Learning Representations_ (2019). Sheng Jia, Jamie Kiros, Jimmy Ba. 
> [[arxiv]](https://arxiv.org/abs/1902.07257) [[openreview]](https://openreview.net/forum?id=HJgd1nAqFX) <br />
# Demo
Trained multitask agent: https://www.youtube.com/watch?v=eGzTDIvX4IY <br/>
Facebook login: https://www.youtube.com/watch?v=IQytRUKmWhs&t=2s

# Requirement
Need to download selenium & install chrome driver for selenium..

# Installation
1. Clone this repo
2. Download MiniWoB++ environment from the original repo https://github.com/stanfordnlp/miniwob-plusplus  <br />
and copy miniwob-plusplus/html folder to miniwob/html in this repo <br />
3. In fact, this html folder could be stored anywhere, but remember to perform one of the following actions: <br />
> * Set environment variable `"WOB_PATH"` to <br />
`file://"your-path-to-miniwob-plusplus"/html/miniwob` <br />
E.g. "your-path-to-miniwob-plusplus" is "/h/sheng/DOM-Q-NET/miniwob
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

# Multitask Assumptions

##  State & Action restrictions
| Item | Maximum number of items |
| ------ | ----------- |
| DOM tree leaves (action space)   | `160` |
| DOM tree |`200`  |
| Instruction tokens    | `16` |

## Attribute embeddings & vocabulary
| Attribute | max vocabulary | Embedding dimension
| ------ | ----------- |----------- |
| Tag   | `100` | `16` |  
| Text (shared with instructions) |`600`  |`48`|
| Class    | `100` | `16` |

> * UNKnown tokens <br />
These are assigned to a random vector such that the cosine distance with the text attribute can yield 1.0 for the direct alignment.

# Acknowledgement
Credit to Dopamine for the implementation of prioritized replay used in dstructs/dopamine_segtree.py   <br />
