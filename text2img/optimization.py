import torch
from torch import nn
from torch.nn import functional as F


class BiObjectiveLoss(nn.Module):
    def __init__(self,
                 criterion_1=F.mse_loss, criterion_2=F.mse_loss,
                 weight_1=1., weight_2=1.):
        """Linear combination of criterions 1 and 2, with weights 1 and 2 respectively.

        Args:
            - criterion_1: callable or nn.Module instance, the first objective
            - criterion_2: callable or nn.Module isntance, the second objective
            - weight_1: float, weight for the first objective
            - weight_2: float, weight for the second objective
        """
        super().__init__()
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.criterion_1 = criterion_1
        self.criterion_2 = criterion_2

    def forward(self, inputs, targets):
        """Given a pair of inputs and their matching pair of targets, compute a weighted sum of
        criterion_1 and criterion_2 with weights weight_1 and weight_2, respectively.

        Args:
            - inputs: tuple or list of torch.Tensor, pair of inputs
            - targets: tuple or list of torch.Tensor, pair of targets

        Returns:
            - tensor, weighted bi-objective loss
        """
        input_1, input_2 = inputs
        target_1, target_2 = targets
        loss_1 = self.criterion_1(input_1, target_1)
        loss_2 = self.criterion_2(input_2, target_2)
        return self.weight_1 * loss_1 + self.weight_2 * loss_2
