from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
import torch.utils.data as data_utils
from tqdm import tqdm

from synapse import Synapse
from synapse.modules import IndexedAccuracyScore, IndexedCrossEntropyLoss


def _create_epoch_desc(epoch_id: int, num_epochs: int):
    max_str_len = len(str(num_epochs))
    epoch_desc = f'[{{:#{max_str_len}d}}|{{}}]'
    return (
        f'{epoch_desc.format(epoch_id, num_epochs)}'
        ' - Training loss {:#.6f}'
        ' - Validation loss {:#.6f}'
        ' - Validation accuracy {:#.6f}'
    )


def train_model(
    dataset: data_utils.Dataset,
    model: nn.Module,
    optimizer: optim.Optimizer,
    synapse: Optional[Synapse] = None,
    task_targets: Optional[List[int]] = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    validation_ratio: float = 0.2,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    model.to(device)
    criterion = IndexedCrossEntropyLoss()

    train_set, validation_set = data_utils.random_split(
        dataset, lengths=[1 - validation_ratio, validation_ratio]
    )
    train_loader = data_utils.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    best_validation_loss = float('inf')
    best_model_state = deepcopy(model.state_dict())

    for epoch_id in range(1, num_epochs + 1):
        epoch_desc = _create_epoch_desc(epoch_id, num_epochs)
        training_losses = []
        model.train()
        for inputs, targets in tqdm(train_loader, desc='Training', leave=False):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, targets, keep_targets=task_targets)
            optimizer.zero_grad()
            if synapse:
                synapse.backward(loss, targets, inputs=inputs)
            else:
                loss.backward()

            nn_utils.clip_grad_value_(model.parameters(), clip_value=2)
            optimizer.step()

            training_losses.append(loss.item())

        training_loss = np.mean(training_losses)
        validation_loss, accuracy = eval_model(
            dataset=validation_set,
            model=model,
            task_targets=task_targets,
            batch_size=batch_size,
            device=device,
        )
        print(epoch_desc.format(training_loss, validation_loss, accuracy))

        if validation_loss < best_validation_loss:
            best_model_state = deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)


def eval_model(
    dataset: data_utils.Dataset,
    model: nn.Module,
    task_targets: Optional[List[int]] = None,
    batch_size: int = 32,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    criterion = IndexedCrossEntropyLoss(reduction='sum')
    metric = IndexedAccuracyScore(reduction='sum')
    evaluation_loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    model.to(device)
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for inputs, targets in tqdm(evaluation_loader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets, keep_targets=task_targets).item()
            total_accuracy += metric(outputs, targets, keep_targets=task_targets).item()

    return (total_loss / len(dataset), total_accuracy / len(dataset))
