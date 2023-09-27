from typing import Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.utils.data as data_utils

from synapse.utils.buffer import ReservoirBuffer

from ._base import Synapse


def _get_grads(params: Iterator[nn.Parameter]):
    return [param.grad for param in params]


def _update_grads(params: Iterator[nn.Parameter], grads: List[torch.Tensor | None]):
    for param, grad in zip(params, grads):
        param.grad = grad


def _flatten_and_dot(
    grads_x: List[torch.Tensor | None], grads_y: List[torch.Tensor | None]
):
    def dot(grad_x: torch.Tensor | None, grad_y: torch.Tensor | None):
        if grad_x is None or grad_y is None:
            return torch.tensor(0)
        return torch.dot(grad_x.flatten(), grad_y.flatten())

    return torch.tensor(
        [dot(grad_x, grad_y) for grad_x, grad_y in zip(grads_x, grads_y)]
    ).sum()


def _project(grads_x: List[torch.Tensor | None], grads_y: List[torch.Tensor | None]):
    corr = _flatten_and_dot(grads_x, grads_y) / _flatten_and_dot(grads_y, grads_y)
    return [
        None if grad_x is None else grad_x - corr * grad_y
        for grad_x, grad_y in zip(grads_x, grads_y)
    ]


class AGEM(Synapse):
    """Implementation of Facebook's
    [Averaged Gradient Episodic Memory](https://arxiv.org/abs/1812.00420)
    using reservoir buffer. PyTorch's implementation referred from the
    [Mammoth](https://github.com/aimagelab/mammoth).

    Args:
        model (nn.Module): Classification model for continual learning
        buffer_size (int, optional): Reservoir buffer size. \
            Defaults to 1024.
        buffer_sample_size (int, optional): Reservoir buffer sample size. \
            Defaults to 32.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1024,
        buffer_sample_size: int = 32,
    ):
        super().__init__(
            model, buffer_size=buffer_size, buffer_sample_size=buffer_sample_size
        )
        self.buffer = ReservoirBuffer(
            self.configs['buffer_size'], self.configs['buffer_sample_size']
        )

    def backward(
        self,
        loss: torch.Tensor,
        targets: torch.LongTensor,
        **model_inputs: torch.Tensor
    ):
        loss.backward()
        if not self.buffer.is_empty():
            loss_grads = _get_grads(self.model.parameters())
            self.model.zero_grad()

            past_items = data_utils.default_collate(self.buffer.sample())
            past_targets = past_items.pop('targets')

            outputs = self.model(**past_items)
            buffer_loss = fn.cross_entropy(outputs, past_targets)
            buffer_loss.backward()
            buffer_grads = _get_grads(self.model.parameters())

            if _flatten_and_dot(loss_grads, buffer_grads).item() < 0:
                new_grad = _project(loss_grads, buffer_grads)
                _update_grads(self.model.parameters(), new_grad)

        self.buffer.add_items(targets=targets, **model_inputs)
