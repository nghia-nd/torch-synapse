from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
)

import torch.utils.data as data_utils


class _HasVisionAttrs(Protocol):
    data: Iterable
    targets: Iterable


class _VisionDataset(data_utils.Dataset, _HasVisionAttrs):
    pass


@dataclass
class Task:
    key: str

    train_set: data_utils.Dataset
    test_set: data_utils.Dataset

    targets: List[int]


class TaskSubset(data_utils.Subset):
    def __init__(
        self,
        dataset: data_utils.Dataset,
        indices: Sequence[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(dataset, indices)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target

    def __repr__(self):
        return f'{self.__class__.__name__}[{len(self)}]'


class TaskDict(Mapping, ABC):
    """An abstract class to split dataset into multiple tasks with separate transforms.
    The task can then be access with the specified key.

    Args:
        root (str): Dataset save root.
        key_to_targets (Dict[str, List[int]]): Task key to targets mapping.
        key_to_transform (Optional[Dict[str, Optional[Callable]]], optional): \
            Task key to data transforms mapping \
            Defaults to None.
        key_to_target_transform (Optional[Dict[str, Optional[Callable]]], optional): \
            Task key to target transforms mapping. \
            Defaults to None.
    """

    @property
    @abstractmethod
    def _train_set(self) -> _VisionDataset:
        pass

    @property
    @abstractmethod
    def _test_set(self) -> _VisionDataset:
        pass

    def __init__(
        self,
        root: str,
        key_to_targets: Dict[str, List[int]],
        key_to_transform: Optional[Dict[str, Optional[Callable]]] = None,
        key_to_target_transform: Optional[Dict[str, Optional[Callable]]] = None,
    ):
        self.root = root
        self.key_to_targets = key_to_targets
        self.key_to_transform = key_to_transform if key_to_transform else {}
        self.key_to_target_transform = (
            key_to_target_transform if key_to_target_transform else {}
        )

        key_to_train_set = self._split_dataset_to_task(self._train_set)
        key_to_test_set = self._split_dataset_to_task(self._test_set)

        self.key_to_task = {
            key: Task(
                key=key,
                train_set=key_to_train_set[key],
                test_set=key_to_test_set[key],
                targets=targets,
            )
            for key, targets in self.key_to_targets.items()
        }

    def _split_dataset_to_task(self, dataset: _VisionDataset):
        target_to_key = {
            target: key
            for key in self.key_to_targets.keys()
            for target in self.key_to_targets[key]
        }

        key_to_point_ids = {key: [] for key in self.key_to_targets.keys()}
        for point_id, target in enumerate(dataset.targets):
            target = int(target)  # mnist has tensor targets
            key_to_point_ids[target_to_key[target]].append(point_id)

        return {
            key: TaskSubset(
                dataset=dataset,
                indices=point_ids,
                transform=self.key_to_transform.get(key),
                target_transform=self.key_to_target_transform.get(key),
            )
            for key, point_ids in key_to_point_ids.items()
        }

    def __getitem__(self, key: str):
        return self.key_to_task[key]

    def __len__(self) -> int:
        return len(self.key_to_task)

    def __iter__(self) -> Iterator[str]:
        return iter(self.key_to_task)

    @property
    def num_targets(self):
        return sum(len(targets) for targets in self.key_to_targets.values())
