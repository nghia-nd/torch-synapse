import pytest
import torch

from synapse.modules import MultiLinear


def test_default_constructor():
    model = MultiLinear(in_features=10)
    assert model.in_features == 10
    assert model.out_features == 0
    assert len(model._task_key_to_linear) == 0


def test_extend_method():
    model = MultiLinear(in_features=10)
    task_key = 'task1'
    out_features = 5
    model.extend(task_key, out_features)
    assert task_key in model._task_key_to_linear
    assert isinstance(model._task_key_to_linear[task_key], torch.nn.Linear)
    assert model.out_features == out_features


def test_forward_concatenation():
    model = MultiLinear(in_features=3)
    model.extend('task1', 2)
    model.extend('task2', 3)

    inputs = torch.randn(1, 3)
    outputs = model(inputs)
    assert outputs.size() == (1, 5)


def test_forward_with_task_key():
    model = MultiLinear(in_features=3)
    model.extend('task1', 2)
    model.extend('task2', 3)

    inputs = torch.randn(1, 3)
    output = model(inputs, 'task1')
    assert output.size() == (1, 2)


def test_forward_with_invalid_task_key():
    model = MultiLinear(in_features=3)
    inputs = torch.randn(1, 3)
    invalid_task_key = 'invalid_task'
    with pytest.raises(KeyError):
        model(inputs, invalid_task_key)
