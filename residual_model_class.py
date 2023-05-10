import torch
import torch.nn.functional as F

"""
Testing whether neural networks with residual blocks improves the prediction capabilities of models
"""


class ResidualLinear(torch.nn.Module):
    def __init__(self, in_nodes: int) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_nodes, in_nodes),
            torch.nn.Linear(in_nodes, in_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualClassifier(torch.nn.Module):
    def __init__(self, in_nodes: int, lr: float = 1e-4) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            ResidualLinear(in_nodes),
            torch.nn.Linear(in_nodes, in_nodes // 4),
            torch.nn.ReLU(),
            ResidualLinear(in_nodes // 4),
            torch.nn.Linear(in_nodes // 4, in_nodes // 16),
            torch.nn.ReLU(),
            ResidualLinear(in_nodes // 16),
            torch.nn.Linear(in_nodes // 16, 2),
        )

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def optimisation_step(self, pred: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        self.optimiser.zero_grad()
        loss = self.loss_func(pred, lab)
        loss.backward()
        self.optimiser.step()

        return loss
