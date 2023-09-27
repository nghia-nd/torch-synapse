import pytest
import torch
import torch.nn as nn

from synapse.modules import IndexedCrossEntropyLoss


def test_correct_output():
    loss_fn = IndexedCrossEntropyLoss()
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([0, 1])
    loss = loss_fn(logits, targets)

    reference_loss_fn = nn.CrossEntropyLoss()
    reference_loss = reference_loss_fn(logits, targets)
    assert loss == reference_loss


def test_specify_keep_targets():
    loss_fn = IndexedCrossEntropyLoss()
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([0, 2])
    keep_targets = [0, 2]
    loss = loss_fn(logits, targets, keep_targets=keep_targets)

    reference_loss_fn = nn.CrossEntropyLoss()
    reference_logits = torch.tensor([[0.2, 0.3], [0.7, 0.2]])
    reference_targets = torch.tensor([0, 1])
    reference_loss = reference_loss_fn(reference_logits, reference_targets)

    assert loss == reference_loss


def test_non_unique_keep_targets():
    loss_fn = IndexedCrossEntropyLoss()
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([0, 1])
    keep_targets = [0, 0, 1]
    with pytest.raises(ValueError):
        loss_fn(logits, targets, keep_targets=keep_targets)


def test_non_categorical_targets():
    loss_fn = IndexedCrossEntropyLoss()
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    floating_point_targets = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
    with pytest.raises(ValueError):
        loss_fn(logits, floating_point_targets)
