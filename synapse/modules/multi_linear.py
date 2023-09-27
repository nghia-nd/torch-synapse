from typing import Optional

import torch
import torch.nn as nn


class MultiLinear(nn.Module):
    """Multi-head Linear module.
    This module could be used as a multi-head classifier for continual models.

    Args:
        in_features (int): Size of the input
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = 0
        self._task_key_to_linear = nn.ModuleDict()

    def extend(self, task_key: str, out_features: int, bias: bool = True):
        """Add new head to the module corresponding to the task key

        Args:
            task_key (str): Task identifier.
            out_features (int): Size of the head output.
            bias (bool, optional): Whether to use bias for this head. \
                Defaults to True.
        """
        self._task_key_to_linear[task_key] = nn.Linear(
            self.in_features, out_features, bias=bias
        )
        self.out_features += out_features

    def forward(self, inputs: torch.Tensor, task_key: Optional[str] = None):
        if task_key:
            return self._task_key_to_linear[task_key](inputs)

        outputs = [linear(inputs) for linear in self._task_key_to_linear.values()]
        return torch.concat(outputs, dim=1)
