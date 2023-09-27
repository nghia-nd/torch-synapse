import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from synapse import Synapse, TaskDict

from ..utils.learning import eval_model, train_model


def train_model_continual(
    task_dict: TaskDict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    synapse: Synapse,
    lr: float = 0.01,
    num_epochs: int = 10,
    batch_size: int = 32,
    validation_ratio: float = 0.2,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    for task in task_dict.values():
        print(task)

    for task in task_dict.values():
        print('=' * 64)
        print(f'Learning task {task.key}')
        model.extend(task.key, len(task.targets))
        train_model(
            dataset=task.train_set,
            model=model,
            optimizer=optimizer,
            synapse=synapse,
            task_targets=task.targets,
            num_epochs=num_epochs,
            batch_size=batch_size,
            validation_ratio=validation_ratio,
            device=device,
        )

        print('\nTEST:')
        task_keys = list(task_dict.keys())
        for key in task_keys[: task_keys.index(task.key) + 1]:
            test_loss, test_accuracy = eval_model(
                dataset=task_dict[key].test_set,
                model=model,
                task_targets=task_dict[key].targets,
                batch_size=batch_size,
                device=device,
            )
            print(
                f'Test task {key}: '
                f'Loss: {test_loss:#.6f} - Accuracy: {test_accuracy:#.6f}'
            )

    print('=' * 64)
    test_set = data_utils.ConcatDataset([task.test_set for task in task_dict.values()])
    test_loss, test_accuracy = eval_model(
        test_set, model=model, batch_size=batch_size, device=device
    )
    print(f'Test all task: ' f'Loss: {test_loss:#.6f} - Accuracy: {test_accuracy:#.6f}')
