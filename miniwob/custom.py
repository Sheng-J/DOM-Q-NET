import ipdb

# Follows previous work Liu et.al 18, and preprocesses months for
# choose date task
transform_dict = {
        "1": "January",
        "2": "February",
        "3": "March",
        "4": "April",
        "5": "May",
        "6": "June",
        "7": "July",
        "8": "August",
        "9": "September",
        "10": "October",
        "11": "November",
        "12": "December"
        }

def convert_goal(goal):
    # goal = goal.replace(".", " ")
    for item in ['.', ',', '"', "/"]:
        goal = goal.replace(item, ' ')
    res = []
    for x in goal.strip().split():
        if x.isdigit():
            x = str(int(x))
        res.append(x)
    #res = res[1:]
    #if res[0] in transform_dict:
    #    res[0] = transform_dict[res[0]]
    
    #NOTE TODO PUT THIS BACK IF choose_date NOT WORKING 
    if res[1] in transform_dict:
        res[1] = transform_dict[res[1]]
    return res


# FIX appending to static variable which could affect other instantiations
class TaskCustomizer(object):
    min_kept_attrs = ["ref", "tag", "text", "classes", "focused", "tampered", "raw_text", "top"]
    debug_attrs = ["tag", "text", "classes", "focused", "tampered", "raw_text"]
    tree_debug_attrs = ["children", "tag", "text", "classes", "focused", "tampered", "depth", "Q", "h", "sim", "adj_V", "raw_text", "top"]

    def __init__(self, attr_vocabs):
        self._attr_vocabs = attr_vocabs
        self.min_kept_attrs.append("is_leaf")
        self.min_kept_attrs.append("adj_V")
        self._leaf_idx = None
        self._raw_idx = None
        self.convert_goal = convert_goal

    def convert_doms(self, doms):
        """
        From doms, create a deep filtered copy, so it wont affect the original
        dom dictionary, which could be used for debugging
        """
        # ipdb.set_trace()
        text_exist_f = lambda x: x.get('text') is not None
        text_non_empty_f = lambda x: x.get('text') != ''
        dom_vals = {key: [] for key in self.min_kept_attrs}
        leaf_vals = {key: [] for key in self.min_kept_attrs}
        filter_f = lambda x: text_exist_f(x) and text_non_empty_f(x)
        for x in doms:
            for key in self.min_kept_attrs:
                #if key == "raw_text":
                #    ipdb.set_trace()
                dom_vals[key].append(get_or_na(x, key))
                #if key == "raw_text":
                #    ipdb.set_trace()
                # if x["is_leaf"] and filter_f(x):
                if x["is_leaf"]:
                    leaf_vals[key].append(get_or_na(x, key))
        return dom_vals, leaf_vals

    def debug_msg(self, dom_elems, dom_index):
        dom_attrs = [str(dom_elems[attr][dom_index]) for attr in self.debug_attrs]
        dom_attrs.insert(0, dom_index)
        debug_msg = ", ".join([attr+"=%s" for attr in self.debug_attrs])
        print(("CLICK DOM %d with"+debug_msg) % tuple(dom_attrs))

    def format_doms(self, dom_vals):
        """
        reformat attrs based doms
        """
        num_doms = len(dom_vals["ref"])
        doms = {idx:{key: dom_vals[key][idx] for key in self.debug_attrs}
                for idx in range(num_doms)}
        return doms

    def tree_format_dom(self, dom, converted_doms, doms_info, leaves_info):
        """
        For debugging purpose, check whether the actual token exists in dataset
        In-place change top dom representation
        """
        self._leaf_idx = 0
        self._raw_idx = 0
        depth = 0
        # ipdb.set_trace()
        self._tree_format(dom, depth, converted_doms, doms_info, leaves_info)

    def _tree_format(self, dom, depth, converted_doms, doms_info, leaves_info):
        depth += 1
        dom["depth"] = depth
        dom_keys = [key for key in dom.keys()]
        # Filter non-debugging attrs
        # Conversion
        for attr in converted_doms.keys():
            if attr in self.tree_debug_attrs:
                dom[attr] = converted_doms[attr][self._raw_idx]
                if attr in self._attr_vocabs:
                    dom[attr] = self._attr_vocabs[attr].mask_unk(dom[attr])

        # Extra debugging info from outside
        for attr in doms_info:
            if attr in self.tree_debug_attrs:
                dom[attr] = doms_info[attr][self._raw_idx]
        if "is_leaf" not in dom:
            ipdb.set_trace()

        if dom["is_leaf"]:
            # Extra debugging info from outside
            for attr in leaves_info:
                if attr in self.tree_debug_attrs:
                    dom[attr] = leaves_info[attr][self._leaf_idx]
            dom["leaf_idx"] = self._leaf_idx
            self._leaf_idx += 1
        dom["raw_idx"] = self._raw_idx
        self._raw_idx += 1
        # recursion!
        for dom_child in dom["children"]:
            self._tree_format(dom_child, depth, converted_doms, doms_info, leaves_info)
        if dom["is_leaf"]:
            dom.pop("children") # for cleaness
        for attr in dom_keys:
            if (attr not in self.tree_debug_attrs):
                dom.pop(attr)

def get_or_na(dom, key):
    val = dom.get(key)
    if val is None:
        val = "NA"
    if key in ["tag", "text", "classes"]:
        if val == "":
            return "NA"

    if key == "focused" or key == "tampered":
        if (val == "NA") or (not val):
            val = [1.0, 0.0]
        else:
            val = [0.0, 1.0]
        # TODO FIX SOLN for Environment Error for SUBMIT BUTTON, ALWAYS non-focused
        if ("tag" in dom) and ("text" in dom): 
            if dom["tag"] == "BUTTON" and dom["text"] == "Submit":
                val = [1.0, 0.0]
        elif dom["tag"] == "BODY":
            val = [1.0, 0.0]
    if key == "top":
        # val = [val/100.0]
        val = [dom["ref"]/10.]
    return val


def get_or_na_default(dom, key):
    val = dom.get(key)
    if val is None:
        val = "NA"
    if key in ["tag", "text", "classes"]:
        if val == "":
            return "NA"

    if key == "focused" or key == "tampered":
        if (val == "NA") or (not val):
            val = [1.0, 0.0]
        else:
            val = [0.0, 1.0]
    return val


def create_customizer(custom_mode, attr_vocabs):
    return TaskCustomizer(attr_vocabs)


