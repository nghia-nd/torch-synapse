import pytest
import torch
import torch.nn as nn
import torch.nn.functional as fn

from synapse.rehearsal import ER


def test_er_init():
    model = nn.Linear(10, 2)
    er = ER(model)
    assert er.buffer.is_empty()


def test_er_invalid_method():
    model = nn.Linear(10, 2)
    with pytest.raises(ValueError):
        ER(model, method='invalid_method')


def test_er_custom_method_without_loss_fn():
    model = nn.Linear(10, 2)
    with pytest.raises(ValueError):
        ER(model, method='custom')


def test_er_backward_update_buffer():
    model = nn.Linear(10, 2)
    er = ER(model)
    targets = torch.tensor([0, 1, 0, 1, 0])
    model_inputs = {'input': torch.randn(5, 10)}
    loss = torch.tensor(0.0, requires_grad=True)
    er.backward(loss, targets, **model_inputs)
    assert len(er.buffer) == 5
    assert 'input' in er.buffer[0] and 'targets' in er.buffer[0]


@pytest.mark.parametrize('method', ['er', 'der', 'der++', 'custom'])
def test_er_backward_success(method):
    model = nn.Linear(10, 2)
    er = ER(
        model, method=method, buffer_sample_size=5, custom_loss_fn=lambda x, y, z: 0
    )
    er.buffer.add_items(
        targets=torch.tensor([1, 1, 1, 1, 1]),
        logits=torch.rand(5, 2),
        input=torch.rand(5, 10),
    )
    targets = torch.tensor([1, 1, 1, 1, 1])
    inputs = torch.rand(5, 10, requires_grad=True)
    er.backward(fn.cross_entropy(model(inputs), targets), targets=targets, input=inputs)
