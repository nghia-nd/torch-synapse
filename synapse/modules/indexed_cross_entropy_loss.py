from typing import List, Optional

import torch
import torch.nn as nn


class IndexedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross entropy loss only for specified classes."""

    def _index_select(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        keep_targets: List[int] = None,
    ):
        new_logits = logits[:, keep_targets]
        new_targets = torch.full_like(targets, self.ignore_index)
        for new_target_value, old_target_value in enumerate(keep_targets):
            new_targets[targets.eq(old_target_value)] = new_target_value

        return new_logits, new_targets

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        keep_targets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if targets.is_floating_point():
            raise ValueError('`targets` must be categorical (containing class indices)')

        if keep_targets:
            if len(keep_targets) != len(set(keep_targets)):
                raise ValueError('`keep_targets` must contains unique values')
            logits, targets = self._index_select(logits, targets, keep_targets)

        return super().forward(logits, targets)
