from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Synapse(ABC):
    def __init__(self, model: nn.Module, **configs):
        """Base class for Synapse.
        New Synapse implementation should inherit this class. Synapse provide a
        basic wrapper around torch's `.backward()` method to preserve
        model's performance on subsequent classification tasks.

        Args:
            model (nn.Module): Classification model for continual learning
            configs: Configurations for Synapse
        """
        self.model = model
        self.configs = configs

    @abstractmethod
    def backward(
        self,
        loss: torch.Tensor,
        targets: torch.LongTensor,
        **model_inputs: torch.Tensor
    ):
        """Wrapper for `.backward()` method.
        Instead of performing `loss.backward()`, use `synapse.backward(loss, ...)`.

        Args:
            loss (torch.Tensor): Classification loss
            targets (torch.LongTensor): Classification targets
            model_inputs (torch.Tensor): Inputs to the model as kwargs. \
                Note: kwargs must match model `.forward()` method parameters
        """
        pass
