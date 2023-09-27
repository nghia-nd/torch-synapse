import torch
import torch.nn as nn

from ._base import Synapse


def _get_flattened_params(model: nn.Module):
    return torch.cat([param.flatten() for param in model.parameters()]).detach().clone()


def _get_flattened_grads(model: nn.Module):
    return (
        torch.cat([param.grad.flatten() for param in model.parameters()])
        .detach()
        .clone()
    )


class SI(Synapse):
    """Implementation of [Synaptic Intelligence](https://arxiv.org/abs/1703.04200).
    PyTorch's implementation referred from the
    [Mammoth](https://github.com/aimagelab/mammoth).

    Args:
        model (nn.Module): Classification model for continual learning
        model_lr (float, optional): Learning rate associated with the optimizer. \
            Defaults to 0.01. \
            Note: Using SI with LRScheduler may yield unexpected result.
        alpha (float, optional): Weight for the penalty term. \
            Defaults to 0.5.
        xi (float, optional): Damping term to bound regularization strength. \
            Defaults to 0.01.
        checkpoint_per_steps (int, optional): Number of steps till next checkpoint. \
            Defaults to 200.
    """

    def __init__(
        self,
        model: nn.Module,
        model_lr: float = 0.01,
        alpha: float = 0.5,
        xi: float = 0.01,
        checkpoint_per_steps: int = 200,
    ):
        super().__init__(
            model,
            model_lr=model_lr,
            alpha=alpha,
            xi=xi,
            checkpoint_per_steps=checkpoint_per_steps,
        )
        self.old_params = _get_flattened_params(self.model)
        self.big_omega = torch.zeros_like(self.old_params)
        self.small_omega = torch.zeros_like(self.old_params)
        self.num_steps = 0

    def _compute_penalty(self):
        # model params can grow, so we only take the old params
        new_params = _get_flattened_params(self.model)
        params_diff = new_params[: len(self.old_params)] - self.old_params
        return (self.big_omega * params_diff**2).sum()

    def _should_checkpoint(self):
        self.num_steps += 1
        if self.num_steps % self.configs['checkpoint_per_steps'] == 0:
            return True
        return False

    def _checkpoint(self):
        new_params = _get_flattened_params(self.model)

        params_diff = new_params[: len(self.old_params)] - self.old_params
        len_diff = len(new_params) - len(self.old_params)

        self.big_omega += self.small_omega / (params_diff**2 + self.configs['xi'])

        self.old_params = new_params
        self.big_omega = torch.cat(
            [self.big_omega, torch.zeros(len_diff, device=self.big_omega.device)]
        )
        self.small_omega = torch.zeros_like(self.old_params)

    def backward(self, loss: torch.Tensor, *_, **__):
        if self._should_checkpoint():
            self._checkpoint()

        penalty = self.configs['alpha'] * self._compute_penalty()
        loss += penalty
        loss.backward()

        new_grads = _get_flattened_grads(self.model)[: len(self.old_params)]
        self.small_omega += self.configs['model_lr'] * new_grads.clip(max=1) ** 2
