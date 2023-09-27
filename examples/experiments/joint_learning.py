import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from synapse import TaskDict

from ..utils.learning import eval_model, train_model


def train_model_joint(
    task_dict: TaskDict,
    model: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 10,
    batch_size: int = 32,
    validation_ratio: float = 0.2,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    train_set = data_utils.ConcatDataset(
        [task.train_set for task in task_dict.values()],
    )
    test_set = data_utils.ConcatDataset(
        [task.test_set for task in task_dict.values()],
    )

    total_targets = sum(len(task.targets) for task in task_dict.values())
    model.extend('all', total_targets)

    train_model(
        dataset=train_set,
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_ratio=validation_ratio,
        device=device,
    )
    print('=' * 64)
    test_loss, test_accuracy = eval_model(
        test_set, model=model, batch_size=batch_size, device=device
    )
    print(f'Test all task: ' f'Loss: {test_loss:#.6f} - Accuracy: {test_accuracy:#.6f}')
