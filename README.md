Code for DOM-Q-NET: Grounded RL on structured language https://openreview.net/pdf?id=HJgd1nAqFX <br />
Documentation will be continuously updated. <br />
Please download MiniWoB++ environment from the original repo
https://github.com/stanfordnlp/miniwob-plusplus  <br />
and copy miniwob-plusplus/html folder to miniwob/html in this repo <br />
In fact, this html folder containing the environment could be stored anywhere, <br />
but remember to perform one of the following options: <br />
[1] Set environment variable "WOB_PATH" to <br />
"your-path-to-miniwob-plusplus"/html/miniwob <br />
[2] Directly modify the base_url on line 33 of instance.py to  <br />
"your-path-to-miniwob-plusplus"/html/miniwob <br />
In my case, base_url='file:///h/sheng/DOM-Q-NET/miniwob/html/miniwob/' <br />

Credit to Dopamine for dstructs/dopamine_segtree.py   <br />
