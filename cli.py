from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from examples.experiments import (
    train_model_continual,
    train_model_joint,
    train_model_sequential,
)
from examples.task_dicts import CIFAR10TaskDict, CIFAR100TaskDict, MNISTTaskDict
from synapse.gradient import AGEM
from synapse.modules import MultiLinear
from synapse.rehearsal import ER
from synapse.structure import SI


class ContinualModel(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int):
        super().__init__()
        model_init_fn = getattr(models, model_name)
        self.feature_extractor = model_init_fn(num_classes=embedding_dim)
        self.classifier = MultiLinear(in_features=embedding_dim)

    def extend(self, task_key: str, num_classes: int):
        self.classifier.extend(task_key, num_classes)

    def forward(self, inputs: torch.Tensor, task_key: Optional[str] = None):
        features = self.feature_extractor(inputs)
        return self.classifier(features, task_key=task_key)


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_TO_TASK_DICT_INIT = {
    'cifar10': lambda dataset_dir: CIFAR10TaskDict(
        root=dataset_dir,
        key_to_targets={
            '1': [0, 1],
            '2': [2, 3],
            '3': [4, 5],
            '4': [6, 7],
            '5': [8, 9],
        },
    ),
    'cifar100': lambda dataset_dir: CIFAR100TaskDict(
        root=dataset_dir,
        key_to_targets={
            '1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            '2': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            '3': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            '4': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            '5': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            '6': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            '7': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            '8': [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            '9': [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            '10': [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        },
    ),
    'mnist': lambda dataset_dir: MNISTTaskDict(
        root=dataset_dir,
        key_to_targets={
            '1': [0, 1],
            '2': [2, 3],
            '3': [4, 5],
            '4': [6, 7],
            '5': [8, 9],
        },
    ),
}

SYNAPSE_TYPE_TO_SYNAPSE_INIT = {
    'er': lambda model, _: ER(model, method='er'),
    'der': lambda model, _: ER(model, method='der'),
    'der++': lambda model, _: ER(model, method='der++'),
    'agem': lambda model, _: AGEM(model),
    'si': lambda model, lr: SI(model, model_lr=lr),
}

TRAINING_TYPE_TO_TRAIN_FN = {
    'joint': lambda args, task_dict, model, optimizer: train_model_joint(
        task_dict=task_dict,
        model=model,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        device=DEFAULT_DEVICE,
    ),
    'sequential': lambda args, task_dict, model, optimizer: train_model_sequential(
        task_dict=task_dict,
        model=model,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        device=DEFAULT_DEVICE,
    ),
    'continual': lambda args, task_dict, model, optimizer: train_model_continual(
        task_dict=task_dict,
        model=model,
        synapse=SYNAPSE_TYPE_TO_SYNAPSE_INIT[args.synapse_type](model, args.lr),
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_ratio=args.validation_ratio,
        device=DEFAULT_DEVICE,
    ),
}


def main():
    parser = ArgumentParser(
        prog='Synapse CLI', description='A simple CLI to run synapse examples'
    )
    parser.add_argument(
        '--dataset_dir',
        default='datasets',
        help='Directory to save the dataset',
    )
    parser.add_argument(
        '--dataset',
        default='cifar10',
        choices=DATASET_TO_TASK_DICT_INIT.keys(),
        help='Dataset to run the experiment',
    )
    parser.add_argument(
        '--model',
        default='alexnet',
        choices=models.list_models(),
        help='TorchVision model name',
    )
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument(
        '--training_type',
        default='continual',
        choices=TRAINING_TYPE_TO_TRAIN_FN.keys(),
        help='Training type',
    )
    parser.add_argument(
        '--synapse_type',
        default='er',
        choices=SYNAPSE_TYPE_TO_SYNAPSE_INIT.keys(),
        help='Synapse type if using continual training',
    )
    parser.add_argument(
        '--num_epochs', default=10, type=int, help='Number of epochs to train model'
    )
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batch size to train model'
    )
    parser.add_argument(
        '--validation_ratio',
        default=0.2,
        type=int,
        help='Validation ratio to split the dataset',
    )
    args = parser.parse_args()

    model = ContinualModel(args.model, embedding_dim=512)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    task_dict = DATASET_TO_TASK_DICT_INIT[args.dataset](args.dataset_dir)
    train_fn = TRAINING_TYPE_TO_TRAIN_FN[args.training_type]
    train_fn(args=args, task_dict=task_dict, model=model, optimizer=optimizer)


if __name__ == '__main__':
    main()
