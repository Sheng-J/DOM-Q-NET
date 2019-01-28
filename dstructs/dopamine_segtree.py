# Using the following repo
# Credit to https://github.com/google/dopamine segmenttree implementation
#
#@article{DBLP:journals/corr/abs-1812-06110,
#          author    = {Pablo Samuel Castro and Subhodeep Moitra and
#                   Carles Gelada and Saurabh Kumar and Marc G. Bellemare},
#          title     = {Dopamine: {A} Research Framework for Deep Reinforcement Learning},
#          journal   = {CoRR},
#          volume    = {abs/1812.06110},
#          year      = {2018},
#          url       = {http://arxiv.org/abs/1812.06110},
#          archivePrefix = {arXiv},
#          eprint    = {1812.06110},
#          timestamp = {Tue, 01 Jan 2019 15:01:25 +0100},
#          biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1812-06110},
#          bibsource = {dblp computer science bibliography, https://dblp.org}
#          }
#
import math
import random
import numpy as np


class SumTree(object):
  """
  This is stored in a list of numpy arrays:
  self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5] ]

  For conciseness, we allocate arrays as powers of two, and pad the excess
  elements with zero values.
  """

  def __init__(self, capacity):
    if capacity <= 0:
      raise ValueError('Sum tree capacity should be positive. Got: {}'.
                       format(capacity))

    self.nodes = []
    tree_depth = int(math.ceil(np.log2(capacity)))
    level_size = 1
    # Constructing from top to bottom
    for _ in range(tree_depth + 1):
      nodes_at_this_depth = np.zeros(level_size)
      self.nodes.append(nodes_at_this_depth)
      level_size *= 2

    self.max_recorded_priority = 1.0

  def _total_priority(self):
    return self.nodes[0][0]

  def sample(self, query_value=None):
    if self._total_priority() == 0.0:
      raise Exception('Cannot sample from an empty sum tree.')
    if query_value and (query_value < 0. or query_value > 1.):
      raise ValueError('query_value must be in [0, 1].')

    # Sample a value in range [0, R), where R is the value stored at the root.
    query_value = random.random() if query_value is None else query_value
    query_value *= self._total_priority()

    # Traverse from top to bottom
    node_index = 0
    for nodes_at_this_depth in self.nodes[1:]:
      # Compute children of previous depth's node.
      left_child = node_index * 2

      left_sum = nodes_at_this_depth[left_child]
      # Each subtree describes a range [0, a), where a is its value.
      if query_value < left_sum:
        node_index = left_child
      else:
        node_index = left_child + 1
        query_value -= left_sum

    return node_index

  def stratified_sample(self, batch_size):
    if self._total_priority() == 0.0:
      raise Exception('Cannot sample from an empty sum tree.')

    bounds = np.linspace(0., 1., batch_size + 1)
    assert len(bounds) == batch_size + 1
    segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
    # [sample prob values for each stratified segment]
    query_values = [random.uniform(x[0], x[1]) for x in segments]
    # list of values
    return [self.sample(query_value=x) for x in query_values]

  def get(self, node_index):
    # leaf values (priority of node)
    return self.nodes[-1][node_index]

  def set(self, node_index, value):
    if value < 0.0:
      raise ValueError('Sum tree values should be nonnegative. Got {}'.
                       format(value))
    # Only update the max but value has to be provided outside
    self.max_recorded_priority = max(value, self.max_recorded_priority)

    delta_value = value - self.nodes[-1][node_index]

    for nodes_at_this_depth in reversed(self.nodes):
      # Note: Adding a delta leads to some tolerable numerical inaccuracies.
      # Only need to add delta for upper values
      nodes_at_this_depth[node_index] += delta_value
      node_index //= 2

    assert node_index == 0, ('Sum tree traversal failed, final node index '
                             'is not 0.')

