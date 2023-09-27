import torch
import torch.nn as nn
import torch.nn.functional as fn

from synapse.gradient import AGEM


def test_agem_init():
    model = nn.Linear(10, 2)
    agem = AGEM(model)
    assert agem.buffer.is_empty()


def test_agem_backward_update_buffer():
    model = nn.Linear(10, 2)
    agem = AGEM(model)
    targets = torch.tensor([0, 1, 0, 1, 0])
    inputs = torch.randn(5, 10)
    loss = torch.tensor(0.0, requires_grad=True)
    agem.backward(loss, targets, input=inputs)
    assert len(agem.buffer) == 5
    assert 'input' in agem.buffer[0] and 'targets' in agem.buffer[0]


def test_agem_backward_success():
    model = nn.Linear(10, 2)
    agem = AGEM(model, buffer_sample_size=5)
    agem.buffer.add_items(
        targets=torch.tensor([1, 1, 1, 1, 1]),
        input=torch.rand(5, 10),
    )
    targets = torch.tensor([1, 1, 1, 1, 1])
    inputs = torch.rand(5, 10, requires_grad=True)
    agem.backward(
        fn.cross_entropy(model(inputs), targets), targets=targets, input=inputs
    )
