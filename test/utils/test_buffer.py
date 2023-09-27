import torch

from synapse.utils import ReservoirBuffer


def test_add_item():
    buffer = ReservoirBuffer(max_size=100, sample_size=10)
    item = {'data': torch.randn(5)}
    buffer.add_item(**item)
    assert buffer.num_seen_examples == 1


def test_is_full():
    buffer = ReservoirBuffer(max_size=3, sample_size=10)
    assert not buffer.is_full()
    buffer.add_item(data=torch.randn(5))
    buffer.add_item(data=torch.randn(5))
    buffer.add_item(data=torch.randn(5))
    assert buffer.is_full()


def test_is_empty():
    buffer = ReservoirBuffer(max_size=100, sample_size=10)
    assert buffer.is_empty()
    buffer.add_item(data=torch.randn(5))
    assert not buffer.is_empty()


def test_sample():
    buffer = ReservoirBuffer(max_size=100, sample_size=10)
    buffer.add_item(data=torch.randn(5))
    buffer.add_item(data=torch.randn(5))
    buffer.add_item(data=torch.randn(5))
    sampled_items = buffer.sample(size=2)
    assert len(sampled_items) == 2


def test_sample_default_size():
    buffer = ReservoirBuffer(max_size=100, sample_size=10)
    buffer.add_items(data=torch.randn(10, 5))
    sampled_items = buffer.sample()
    assert len(sampled_items) == 10
