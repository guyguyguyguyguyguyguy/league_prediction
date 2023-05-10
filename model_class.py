from collections import OrderedDict
from typing import Iterable, List, NewType, Optional

import numpy.typing as npt
import torch
import torch.nn.functional as F

"""
Simple neural networks used to make predictions over the data
"""


class Classifier(torch.nn.Module):
    def __init__(
        self,
        no_features: int,
        *,
        lr: float = 1e-4,
        repam: str = "l2",
        drop: bool = False,
    ) -> None:
        super().__init__()
        self.net = (
            torch.nn.Sequential(
                torch.nn.Linear(no_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(64, 16),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(16, 2),
            )
            if drop
            else torch.nn.Sequential(
                torch.nn.Linear(no_features, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 2),
            )
        )

        self.loss_func = torch.nn.BCEWithLogitsLoss()

        if repam == "l1":
            self.optimiser = torch.optim.Adam(self.parameters(), lr=lr)
            self.optimisation_step = self._l1_optimisation_step

        else:
            self.optimiser = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=1e-5
            )  # Weight decay is L2-reparamterisation
            self.optimisation_step = self._l2_optimisation_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # Wank
    def _l1_optimisation_step(
        self, pred: torch.Tensor, lab: torch.Tensor
    ) -> torch.Tensor:
        self.optimiser.zero_grad()

        l1 = 0.5 * sum([p.abs().sum() for p in self.parameters()])
        loss = self.loss_func(pred, lab) + l1
        loss.backward()
        self.optimiser.step()

        return loss

    def _l2_optimisation_step(
        self, pred: torch.Tensor, lab: torch.Tensor
    ) -> torch.Tensor:
        self.optimiser.zero_grad()
        loss = self.loss_func(pred, lab)
        loss.backward()
        self.optimiser.step()

        return loss


# Adapted from https://github.com/lancopku/meProp/blob/master/src/pytorch/model.py
class NewClassifier(torch.nn.Module):
    def __init__(
        self,
        no_features: int,
        hidden: List[int] | npt.NDArray,
        *,
        feature_indices: Optional[npt.NDArray] = None,
        lr: float = 1e-4,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.hidden = hidden
        self.feature_indices = feature_indices
        self.dropout = dropout
        self.net = torch.nn.Sequential(self._create(no_features, hidden, dropout))
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=1e-5
        )  # Weight decay is L2-reparamterisation

    def _create(
        self,
        no_features: int,
        hidden: List[int] | npt.NDArray,
        dropout: Optional[float],
    ) -> OrderedDict:
        d = OrderedDict()
        if self.feature_indices is not None:
            no_features = len(self.feature_indices)
        d["linear 0"] = torch.nn.Linear(no_features, hidden[0])
        d["relu 0"] = torch.nn.ReLU()
        if dropout:
            d["dropout 0"] = torch.nn.Dropout(dropout)

        for i, (h_in, h_out) in enumerate(zip(hidden, hidden[1:]), start=1):
            d[f"linear {i}"] = torch.nn.Linear(h_in, h_out)
            d[f"relu {i}"] = torch.nn.ReLU()
            if dropout:
                d[f"dropout {i}"] = torch.nn.Dropout(p=dropout)

        d[f"linear {len(hidden)}"] = torch.nn.Linear(hidden[-1], 2)
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_indices is not None:
            x = x[:, self.feature_indices]
        x = self.net(x)
        if not self.training:
            return F.softmax(x, dim=-1)
        return x

    def optimisation_step(self, pred: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        self.optimiser.zero_grad()
        loss = self.loss_func(pred, lab)
        loss.backward()
        self.optimiser.step()

        return loss
