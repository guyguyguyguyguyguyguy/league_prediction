import einops
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange

"""
Test model to see whehether a convolutinal neural network can capture the complex structure of the data
"""


class ConvCalssifier(torch.nn.Module):
    def __init__(self, lr: float = 1e-4) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 1, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3),
            torch.nn.Conv1d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3),
            Rearrange("b h w -> b (h w)"),
            torch.nn.Linear(776, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=1e-5
        )  # Weight decay is L2-reparamterisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b v -> b 1 v")
        return self.net(x)

    def optimisation_step(self, pred: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        self.optimiser.zero_grad()
        loss = self.loss_func(pred, lab)
        loss.backward()
        self.optimiser.step()

        return loss
