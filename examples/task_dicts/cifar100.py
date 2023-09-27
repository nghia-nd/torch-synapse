from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from synapse import TaskDict


class CIFAR100TaskDict(TaskDict):
    def _create_dataset(self, train: bool):
        return CIFAR100(
            root=self.root,
            train=train,
            download=True,
            transform=Compose(
                [
                    Resize((64, 64), antialias=True),
                    ToTensor(),
                    Normalize(
                        mean=[0.5071, 0.4865, 0.4409],
                        std=[0.2673, 0.2564, 0.2762],
                    ),
                ]
            ),
        )

    @property
    def _train_set(self):
        return self._create_dataset(train=True)

    @property
    def _test_set(self):
        return self._create_dataset(train=False)
