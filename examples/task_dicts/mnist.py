from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from synapse import TaskDict


class MNISTTaskDict(TaskDict):
    def _create_dataset(self, train: bool):
        return MNIST(
            root=self.root,
            train=train,
            download=True,
            transform=Compose(
                [
                    Lambda(lambda image: image.convert('RGB')),
                    Resize((64, 64), antialias=True),
                    ToTensor(),
                    Normalize(mean=(0.1307,), std=(0.3081,)),
                ]
            ),
        )

    @property
    def _train_set(self):
        return self._create_dataset(train=True)

    @property
    def _test_set(self):
        return self._create_dataset(train=False)
