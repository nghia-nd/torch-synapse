from typing import Callable, Literal, get_args

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.utils.data as data_utils

from synapse.utils.buffer import ReservoirBuffer

from ._base import Synapse

_Methods = Literal['er', 'der', 'der++', 'custom']
_CustomLoss = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class ER(Synapse):
    """Implementation of vanilla Experience Replay (store past data in a
    reservoir buffer for future rehearsal) and improved versions:
    [Dark Experience Replay, Dark Experience Replay++](https://arxiv.org/abs/2004.07211)
    You can also specify custom loss function to calculate rehearsal loss.
    PyTorch's implementation referred from the
    [Mammoth](https://github.com/aimagelab/mammoth).

    Args:
        model (nn.Module): Classification model for continual learning
        buffer_size (int, optional): Reservoir buffer size. \
            Defaults to 1024.
        buffer_sample_size (int, optional): Reservoir buffer sample size. \
            Defaults to 32.
        alpha (float, optional): Weight for loss term in ER, DER, and custom loss. \
            Defaults to 1.0.
        beta (float, optional): Weight for 2nd loss term in DER++. \
            Defaults to 1.0.
        method (_Methods, optional): Rehearsal method to use. \
            Defaults to 'der++'.
        custom_loss_fn (_CustomLoss | None, optional): Loss function for 'custom' \
            Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1024,
        buffer_sample_size: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
        method: _Methods = 'der++',
        custom_loss_fn: _CustomLoss | None = None,
    ):
        if method not in get_args(_Methods):
            raise ValueError(f'`method` must be one of {get_args(_Methods)}')
        if method == 'custom' and custom_loss_fn is None:
            raise ValueError(
                'must define a `custom_loss_fn` for `method` == \'custom\''
            )

        super().__init__(
            model,
            buffer_size=buffer_size,
            buffer_sample_size=buffer_sample_size,
            alpha=alpha,
            beta=beta,
            method=method,
            custom_loss_fn=custom_loss_fn,
        )
        self.buffer = ReservoirBuffer(
            self.configs['buffer_size'], self.configs['buffer_sample_size']
        )

    def _augment_loss(
        self,
        logits: torch.Tensor,
        past_logits: torch.Tensor,
        past_targets: torch.Tensor,
    ):
        if self.configs['method'] == 'er':
            return self.configs['alpha'] * fn.cross_entropy(logits, past_targets)
        if self.configs['method'] == 'der':
            return self.configs['alpha'] * fn.mse_loss(logits, past_logits)
        if self.configs['method'] == 'der++':
            return self.configs['alpha'] * fn.cross_entropy(
                logits, past_targets
            ) + self.configs['beta'] * fn.mse_loss(logits, past_logits)
        if self.configs['method'] == 'custom':
            return self.configs['custom_loss_fn'](logits, past_logits, past_targets)

    def backward(
        self,
        loss: torch.Tensor,
        targets: torch.LongTensor,
        **model_inputs: torch.Tensor,
    ):
        with torch.no_grad():
            logits = self.model(**model_inputs)

        if not self.buffer.is_empty():
            past_item_seq = self.buffer.sample()

            # pop past_item in-place, note that past_item's keys has been modified
            past_logit_seq = [item.pop('logits') for item in past_item_seq]

            past_items = data_utils.default_collate(past_item_seq)
            past_targets = past_items.pop('targets')

            logits = self.model(**past_items)

            min_past_logit = min(
                past_logit.min().item() for past_logit in past_logit_seq
            )
            past_logits = torch.full_like(logits, min_past_logit)
            for i, past_logit in enumerate(past_logit_seq):
                past_logits[i, : len(past_logit)] = past_logit

            loss += self._augment_loss(logits, past_logits, past_targets)

        loss.backward()
        self.buffer.add_items(logits=logits, targets=targets, **model_inputs)
