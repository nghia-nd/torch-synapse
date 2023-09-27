from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from synapse import TaskDict


class CIFAR10TaskDict(TaskDict):
    def _create_dataset(self, train: bool):
        return CIFAR10(
            root=self.root,
            train=train,
            download=True,
            transform=Compose(
                [
                    Resize((64, 64), antialias=True),
                    ToTensor(),
                    Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2470, 0.2435, 0.2616],
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
