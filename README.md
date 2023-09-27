# Synapse
[WIP] A Python library to extend image classification model capabilities for continual learning 

## Installation

```
pip install git+https://github.com/nghia-nd/torch-synapse.git
```

## Usage
The library provides a simple wrapper for the `loss.backward()` method to implement many continual learning methods.

A simple training loop in PyTorch resembles the following patterns:
```python
for epoch_id in range(1, num_epochs + 1): 
        model.train()
        for inputs, targets in train_loader: 
            outputs = model(inputs) 
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()	
            optimizer.step()
```
Synapse introduces minimal changes to the existing training loop to implement continual learning:
```python
synapse = Synapse(model, **configs)
for epoch_id in range(1, num_epochs + 1): 
        model.train()
        for inputs, targets in train_loader: 
            outputs = model(inputs) 
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            synapse.backward(loss, targets, inputs=inputs) # instead of loss.backward()
            optimizer.step()	

```

## Continual learning methods
Synapse currently supports 5 methods:
- Experience Replay: `synapse.rehearsal.ER`
- Dark Experience Replay: `synapse.rehearsal.ER`
- Dark Experience Replay ++: `synapse.rehearsal.ER`
```python
class ER(Synapse):
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1024,
        buffer_sample_size: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
        method: _Methods = 'der++',
        custom_loss_fn: _CustomLoss | None = None,
    ):
```
- Average Gradient Episodic Memory: `synapse.gradient.AGEM`
```python
class AGEM(Synapse):
    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1024,
        buffer_sample_size: int = 32,
    ):
```
- Synaptic Intelligence: `synapse.structure.SI`
```python
class SI(Synapse):
    def __init__(
        self,
        model: nn.Module,
        model_lr: float = 0.01,
        alpha: float = 0.5,
        xi: float = 0.01,
        checkpoint_per_steps: int = 1250,
    ):
```
Credit: Many of these implementations are referred from the [Mammoth](https://github.com/aimagelab/mammoth).

## Utilities
Synapse also provides a simple wrapper for `torch.utils.data.Dataset` to split a dataset into multiple tasks:
```python
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from synapse import TaskDict

class CIFAR100TaskDict(TaskDict):
    def _create_dataset(self, train: bool):
        return CIFAR100(root=self.root, train=train, download=True, ...)

    @property
    def _train_set(self):
        return self._create_dataset(train=True)

    @property
    def _test_set(self):
        return self._create_dataset(train=False)

cifar10_task_dict = CIFAR10TaskDict(
      root='datasets',
      key_to_targets={
          '1': [0, 1],
          '2': [2, 3],
          '3': [4, 5],
          '4': [6, 7],
          '5': [8, 9],
      },
  )
```

## Examples
Synapse also provides some examples of using the library in the `examples/` folder. There are also a CLI to quickly run these examples:
```
python cli.py --help
usage: Synapse CLI [-h] [--dataset_dir DATASET_DIR] [--dataset {cifar10,cifar100,mnist}]
                   [--model {...}]
                   [--lr LR] [--training_type {joint,sequential,continual}] [--synapse_type {er,der,der++,agem,si}] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                   [--validation_ratio VALIDATION_RATIO]

A simple CLI to run synapse examples

options:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
                        Directory to save the dataset
  --dataset {cifar10,cifar100,mnist}
                        Dataset to run the experiment
  --model {...}
                        TorchVision model name
  --lr LR               Learning rate
  --training_type {joint,sequential,continual}
                        Training type
  --synapse_type {er,der,der++,agem,si}
                        Synapse type if using continual training
  --num_epochs NUM_EPOCHS
                        Number of epochs to train model
  --batch_size BATCH_SIZE
                        Batch size to train model
  --validation_ratio VALIDATION_RATIO
                        Validation ratio to split the dataset
```
