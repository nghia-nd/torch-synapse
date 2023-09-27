import pytest
import torch

from synapse.modules import IndexedAccuracyScore


def test_correct_output():
    score = IndexedAccuracyScore()
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([1, 1])
    result = score(logits, targets)
    assert result.item() == 0.5


def test_specify_top_k():
    score = IndexedAccuracyScore(top_k=2)
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([2, 2])
    result = score(logits, targets)
    assert result.item() == 1.0


def test_reduction_sum():
    score = IndexedAccuracyScore(reduction='sum')
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([1, 0])
    result = score(logits, targets)
    assert result.item() == 2.0


def test_reduction_none():
    score = IndexedAccuracyScore(reduction='none')
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([1, 1])
    result = score(logits, targets)
    assert torch.equal(result, torch.tensor([1, 0]))


def test_keep_targets():
    score = IndexedAccuracyScore()
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2], [0.7, 0.1, 0.2]])
    targets = torch.tensor([2, 1, 0])
    keep_targets = [0, 2]
    result = score(logits, targets, keep_targets=keep_targets)
    assert result.item() == 1.0


def test_invalid_reduction():
    with pytest.raises(ValueError):
        IndexedAccuracyScore(reduction='invalid_reduction')


def test_non_unique_keep_targets():
    score = IndexedAccuracyScore()
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]])
    targets = torch.tensor([0, 1])
    keep_targets = [0, 0, 1]
    with pytest.raises(ValueError):
        score(logits, targets, keep_targets=keep_targets)
