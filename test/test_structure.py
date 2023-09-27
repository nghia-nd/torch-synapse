import torch
import torch.nn as nn
import torch.nn.functional as fn

from synapse.structure import SI


def test_si_init():
    model = nn.Linear(10, 2)
    si = SI(model)
    assert all(si.big_omega.eq(0))
    assert all(si.small_omega.eq(0))
    assert si.num_steps == 0


def test_si_should_checkpoint():
    model = nn.Linear(10, 2)
    si = SI(model, checkpoint_per_steps=200)
    for i in range(1, 201):
        if i % 200 == 0:
            assert si._should_checkpoint() is True
        else:
            assert si._should_checkpoint() is False


def test_si_backward_success():
    model = nn.Linear(10, 2)
    si = SI(model, checkpoint_per_steps=1)
    for _ in range(2):
        inputs = torch.rand(5, 10, requires_grad=True)
        targets = torch.tensor([1, 1, 1, 1, 1])
        si.backward(
            fn.cross_entropy(model(inputs), targets),
        )
