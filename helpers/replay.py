import torch
import numpy as np
import random
from .config import device


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
  e = 0.001
  a = 0.6
  beta = 0.4
  beta_increment_per_sampling = 0.000001

  def __init__(self, capacity):
    self.tree = SumTree(capacity)
    self.capacity = capacity

  def _get_priority(self, error):
    return (np.abs(error) + self.e) ** self.a

  @property
  def n_entries(self):
    return self.tree.n_entries

  def add(self, error, sample):
    p = self._get_priority(error)
    self.tree.add(p, sample)

  def sample(self, n):
    batch = []
    idxs = []
    segment = self.tree.total() / n
    priorities = []

    self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

    for i in range(n):
      a = segment * i
      b = segment * (i + 1)

      #patch - sometimes returns unexisting nodes (data = 0)
      data = 0
      while data == 0:
        s = random.uniform(a, b)
        (idx, p, data) = self.tree.get(s)

      priorities.append(p)
      batch.append(data)
      idxs.append(idx)

    sampling_probabilities = priorities / self.tree.total()
    is_weight = np.power(self.tree.n_entries * sampling_probabilities,
                         -self.beta)
    is_weight /= is_weight.max()
    is_weight = torch.from_numpy(is_weight).float().to(device)

    states = torch.from_numpy(
        np.vstack([e[0] for e in batch if e is not None])).float().to(
        device)
    colleague_states = torch.from_numpy(
        np.vstack([e[1] for e in batch if e is not None])).float().to(
        device)
    actions = torch.from_numpy(
        np.vstack([e[2] for e in batch if e is not None])).float().to(
        device)
    rewards = torch.from_numpy(
        np.vstack([e[3] for e in batch if e is not None])).float().to(
        device)
    next_states = torch.from_numpy(np.vstack(
        [e[4] for e in batch if e is not None])).float().to(
        device)
    colleague_next_states = torch.from_numpy(np.vstack(
        [e[5] for e in batch if e is not None])).float().to(
        device)
    dones = torch.from_numpy(
        np.vstack([e[6] for e in batch if e is not None]).astype(
            np.uint8)).float().to(device)

    return (states,
            colleague_states,
            actions,
            rewards,
            next_states,
            colleague_next_states,
            dones),\
           idxs,\
           is_weight

  def update(self, idx, error):
    p = self._get_priority(error)
    self.tree.update(idx, p)


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
