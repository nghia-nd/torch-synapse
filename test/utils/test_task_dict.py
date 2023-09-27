import torch.utils.data as data_utils

from synapse.utils import TaskDict


def test_task_dict():
    class TrainDataset(data_utils.Dataset):
        def __init__(self):
            self.data = [10, 10, 10, 10]
            self.targets = [0, 1, 2, 3]

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    class TestDataset(data_utils.Dataset):
        def __init__(self):
            self.data = [10, 10]
            self.targets = [0, 3]

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    class TestTaskDict(TaskDict):
        @property
        def _train_set(self):
            return TrainDataset()

        @property
        def _test_set(self):
            return TestDataset()

    key_to_targets = {'task1': [0, 1], 'task2': [2, 3]}
    task_dict = TestTaskDict(root='', key_to_targets=key_to_targets)
    assert len(task_dict) == 2
    assert task_dict['task1'].key == 'task1'
    assert task_dict['task2'].key == 'task2'
    assert task_dict.num_targets == 4

    assert len(task_dict['task1'].train_set) == 2
    assert len(task_dict['task1'].test_set) == 1

    assert len(task_dict['task1'].train_set) == 2
    assert len(task_dict['task1'].test_set) == 1
