from copy import deepcopy

import numpy as np
import torch
import torch.utils.data as data_utils


class ReservoirBuffer(data_utils.Dataset):
    def __init__(self, max_size: int, sample_size: int):
        """Simple resevoir buffer implementation that can act as a dataset.

        Args:
            max_size (int): Buffer max size
            sample_size (int): Buffer sample size
        """
        self.max_size = max_size
        self.sample_size = sample_size
        self.num_seen_examples = 0
        self._examples = []

    def __len__(self):
        return min(self.max_size, self.num_seen_examples)

    def __iter__(self):
        return iter(self._examples)

    def __getitem__(self, index: int):
        return self._examples[index]

    def is_full(self):
        return len(self._examples) == self.max_size

    def is_empty(self):
        return len(self._examples) == 0

    def _choose_item_index(self):
        index = np.random.randint(self.num_seen_examples + 1)
        return index if index < self.max_size else None

    def add_item(self, **item: torch.Tensor):
        if not self.is_full():
            self._examples.append(item)
        else:
            index = self._choose_item_index()
            if index:
                self._examples[index] = item

        self.num_seen_examples += 1

    def add_items(self, **batched_items: torch.Tensor):
        item_keys = batched_items.keys()
        for item_values in zip(*batched_items.values()):
            item = {key: value.detach() for key, value in zip(item_keys, item_values)}
            self.add_item(**item)

    def sample(self, size: int | None = None):
        if not size:
            size = self.sample_size
        num_choices = len(self._examples)
        size = min(size, num_choices)

        return deepcopy(np.random.choice(self._examples, size=size, replace=False))
