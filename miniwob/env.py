import collections
import time
from miniwob.instance import MiniWoBInstance
import ipdb
from pprint import pprint


class MiniWoBEnvironment:
    default_kept_attrs = ["ref", "tag", "text", "classes"]

    def __init__(self, task_name, customizer):
        self._instance = MiniWoBInstance(task_name+".html")
        self._customizer = customizer
        self._epi_step = 0
        self._epi_reward = 0
        self._curr_doms = None
        self._curr_goal = None
        self._curr_top_dom = None
        self._curr_leaves = None

    @property
    def epi_step(self):
        return self._epi_step

    @property
    def epi_reward(self):
        return self._epi_reward

    @property
    def is_done(self):
        return self._instance.is_done["done"]

    def reset(self, seed=None):
        # TODO
        self._instance.force_stop()
        self._instance.begin_task(seed)
        self._curr_goal = None
        self._curr_doms, self._curr_goal, self._curr_leaves, self._curr_top_dom = self._get_env_state()
        self._epi_step = 0
        self._epi_reward = 0
        item_tuple = (self._curr_doms, self._curr_goal, self._curr_leaves)
        # ipdb.set_trace()
        return item_tuple

    def reset_show(self, seed=None):
        res = self.reset()
        print(self._curr_goal)
        print('S(t=%d): Tree' % self._epi_step)
        self._customizer.tree_format_dom(self._curr_top_dom, self._curr_doms, {}, {})
        pprint(self._curr_top_dom)
        return res

    def step_show(self, dom_index__act_type__text_index):
        print("A(t=%d): " % self._epi_step)
        self._customizer.debug_msg(self._curr_leaves, dom_indedom_index__act_type__text_index)
        res_tuple, reward, done, info = self.step(dom_index__act_type__text_index)
        print("R(t=%d) = %d" % (self._epi_step, reward))
        if not done:
            print(self._curr_goal)
            print('S(t=%d): Tree' % self._epi_step)
            self._customizer.tree_format_dom(self._curr_top_dom, self._curr_doms, {}, {})
            pprint(self._curr_top_dom)
        time.sleep(2)
        return res_tuple, reward, done, info


    def is_valid_step(self, dom_index):
        return (dom_index < len(self._curr_leaves["ref"]) and dom_index >= 0)

    def debug_step(self, dom_index__act_type__text_index, doms_info, leaves_info):
        print(self._curr_goal)
        print('S(t=%d): Tree' % self._epi_step)
        self._customizer.tree_format_dom(self._curr_top_dom, self._curr_doms, doms_info, leaves_info)
        pprint(self._curr_top_dom)
        print("A(t=%d): " % self._epi_step)
        self._customizer.debug_msg(self._curr_leaves, dom_index__act_type__text_index)
        res_tuple, reward, done, info = self.step(dom_index__act_type__text_index)
        print("R(t=%d) = %d" % (self._epi_step, reward))
        return res_tuple, reward, done, info

    def step(self, dom_index__act_type__text_index):
        """
        dom_index:  0 ~ len(leaves)-1
        act_type:   0 or 1
        text_index: 0~len(goal_tokens)-1
        """
        dom_index, act_type, text_index = dom_index__act_type__text_index
        dom_ref = self._curr_leaves["ref"][dom_index]
        self._epi_step += 1
        # NOTE formmode CHANGE HAPPEND HERE
        if act_type == 0:
            self._instance.dom_click(dom_ref)
        elif act_type == 1:
            form_text = self._curr_goal[text_index]
            if form_text == "<pad>":
                ipdb.set_trace()
            self._instance.focus_and_type(dom_ref, form_text)
        else:
            raise ValueError("not valid act type")
        self._curr_doms, _, self._curr_leaves, self._curr_top_dom = self._get_env_state()
        metadata = self._instance.metadata
        reward, done = metadata["raw_reward"], metadata["done"]
        # info = metadata["info"]
        # assert int(reward) == 1 or int(reward) == 0 or int(reward)==-1
        if int(reward) == -1:
            reward = 0
        self._epi_reward += reward
        item_tuple = (self._curr_doms, self._curr_goal, self._curr_leaves), reward, done, _ 
        return item_tuple

    def _get_env_state(self):
        dom_elems = []
        # in-place dom flattening
        # Implicitly add "adj_V", "is_leaf"
        # if self._curr_goal is None:
        goal = self._customizer.convert_goal(self._instance.utterance)
        self._curr_goal = goal
        #else:
        #    goal = self._curr_goal
        top_dom = self._instance.dom
        top_dom["adj_V"] = []
        flatten_dom(top_dom, dom_elems)
        # dom_elems is a in-place flattened representation of top_dom
        dom_vals, leaf_vals = self._customizer.convert_doms(dom_elems)
        return (dom_vals, goal, leaf_vals, top_dom)


def flatten_dom(dom_elem, dom_list):
    dom_list.append(dom_elem)
    dom_elem_idx = dom_list.index(dom_elem)
    dom_elem["is_leaf"] = (len(dom_elem["children"]) == 0)
    for i, child_elem in enumerate(dom_elem["children"]):
        dom_list[dom_elem_idx]["adj_V"].append(len(dom_list))
        child_elem["adj_V"] = [dom_elem_idx]
        flatten_dom(child_elem, dom_list)


def filter_leaves(iterable, dom_num_children):
    items = []
    for item, num_children in zip(iterable, dom_num_children):
        if num_children == 0:
            items.append(item)
    return items


transform_dict = {
        "January": "1",
        "February": "2",
        "March": "3",
        "April": "4",
        "May": "5",
        "June": "6",
        "July": "7",
        "August": "8",
        "September": "9",
        "October": "10",
        "November": "11",
        "December": "12"
        }
