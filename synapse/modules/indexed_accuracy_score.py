from typing import List, Literal, Optional, get_args

import torch
import torch.nn as nn

_Reduction = Literal['mean', 'sum', 'none']


class IndexedAccuracyScore(nn.Module):
    """Accuracy score only for specified classes.
    Please note that this class disable gradient calculation on the output.

    Args:
        top_k (int, optional): Using top-k accuracy.
            Defaults to 1.
        ignore_index (int, optional): Ignoring the specified row from the calculation.
            Defaults to -100.
        reduction (_Reduction, optional): Dimensionality reduction on the output.
            Defaults to 'mean'.
    """

    def __init__(
        self, top_k: int = 1, ignore_index: int = -100, reduction: _Reduction = 'mean'
    ):
        if reduction not in get_args(_Reduction):
            raise ValueError(f'`reduction` must be one of {get_args(_Reduction)}')

        super().__init__()
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.reduction = reduction

    def _index_select(
        self, logits: torch.Tensor, targets: torch.Tensor, keep_targets: List[int]
    ):
        new_logits = logits[:, keep_targets]
        new_targets = torch.full_like(targets, self.ignore_index)
        for new_target_value, old_target_value in enumerate(keep_targets):
            new_targets[targets.eq(old_target_value)] = new_target_value
        return new_logits, new_targets

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        keep_targets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if keep_targets:
            if len(keep_targets) != len(set(keep_targets)):
                raise ValueError('`keep_targets` must contains unique values')
            logits, targets = self._index_select(logits, targets, keep_targets)

        _, top_k_logit_indexes = logits.topk(k=self.top_k, dim=-1)
        top_k_is_equal = top_k_logit_indexes.eq(targets.unsqueeze(dim=-1))
        is_equal = top_k_is_equal.sum(dim=-1)

        not_ignored_is_equal = is_equal[~targets.eq(self.ignore_index)]

        if self.reduction == 'mean':
            return not_ignored_is_equal.mean(dtype=float)
        if self.reduction == 'sum':
            return not_ignored_is_equal.sum(dtype=float)
        return not_ignored_is_equal
