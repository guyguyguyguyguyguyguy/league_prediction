import statistics
from typing import Tuple

import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from make_dataset import creat_dataframe, make_loader, seperate_data

"""
Using traditional non-graph based neural networks to carry out classification of the winner give the two players stats.
"""


class EmbeddingAE(nn.Sequential):
    def __init__(self, input_size: int) -> None:
        super().__init__(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 8),
            nn.ReLU(),
        )
        self.embedding_size = input_size // 8


class PredictionModel(torch.nn.Sequential):
    def __init__(self, input_size: int) -> None:
        super().__init__(
            nn.Linear(input_size, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if not self.training:
            return F.softmax(x, dim=-1)
        return x


class PlayerDataL(Dataset):
    def __init__(self, data: pl.DataFrame) -> None:
        super().__init__()
        self.player1, self.player2, self.label = self.seperate_player_data(data)

    def seperate_player_data(self, data: pl.DataFrame) -> Tuple[pl.DataFrame, ...]:
        players_data, label_data = seperate_data(data)

        p1_data = players_data.select(pl.col("^blue.*$"))
        p2_data = players_data.select(pl.col("^red.*$"))

        return p1_data, p2_data, label_data

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        p1_row = torch.tensor(
            self.player1[idx].to_numpy(), dtype=torch.float32
        ).squeeze()
        p2_row = torch.tensor(
            self.player2[idx].to_numpy(), dtype=torch.float32
        ).squeeze()
        label_row = torch.tensor(
            self.label[idx].to_numpy(), dtype=torch.float32
        ).squeeze()

        return (p1_row, p2_row), label_row


def make_embedding_dataloaders(df: pl.DataFrame) -> Tuple[DataLoader, ...]:
    train_dataset = PlayerDataL(df)
    train_dataload, val_dataload = make_loader(
        *random_split(
            train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        ),
        batch_size=512,
    )

    return train_dataload, val_dataload


def _train(
    e_m: EmbeddingAE,
    p_m: PredictionModel,
    p1: torch.Tensor,
    p2: torch.Tensor,
    ls: torch.Tensor,
    optimiser: torch.optim.Adam,
    loss_fn: torch.nn.BCEWithLogitsLoss,
) -> float:
    optimiser.zero_grad()
    embeds = e_m(p1), e_m(p2)
    preds = p_m(torch.hstack([*embeds]))
    loss = loss_fn(preds, ls)
    loss.backward()
    optimiser.step()

    return loss.item()


# TODO: This is not learning!
def train_models(
    e_m: EmbeddingAE,
    p_m: PredictionModel,
    data: DataLoader,
    device: torch.device,
    lr_e: float = 1e-4,
    lr_p: float = 1e-4,
) -> float:
    optimiser = torch.optim.Adam(
        [
            {"params": e_m.parameters(), "lr": lr_p},
            {"params": p_m.parameters(), "lr": lr_e},
        ],
        weight_decay=1e-5,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    e_m.to(device)
    p_m.to(device)
    e_m.train()
    p_m.train()

    losses = []
    for (p1, p2), ls in data:
        p1 = p1.to(device)
        p2 = p2.to(device)
        loss = _train(e_m, p_m, p1, p2, ls, optimiser, loss_fn)
        losses.append(loss)
        # loss = _train(e_m, p_m, p2, p1, ls.flip(dims=[-1]), optimiser, loss_fn)
        # losses.append(loss)

    return statistics.mean(losses)


def val_prediction(
    e_model: EmbeddingAE,
    p_model: PredictionModel,
    val_data: DataLoader,
    device: torch.device,
) -> float:
    e_model.to(device)
    p_model.to(device)
    e_model.eval()
    p_model.eval()

    with torch.no_grad():
        val_acc = []
        for (p1, p2), l in val_data:
            embeds = e_model(p1), e_model(p2)
            pred = p_model(torch.hstack([*embeds]))
            val_acc.append(
                (pred.argmax(axis=1) == l.argmax(axis=1))
                .to(torch.float32)
                .cpu()
                .mean()
                .item()
            )
    return statistics.mean(val_acc)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = creat_dataframe()
    df = df.sample(50000)
    train_data, val_data = make_embedding_dataloaders(df)

    embed_model = EmbeddingAE(388)
    pred_model = PredictionModel(embed_model.embedding_size * 2)

    print("Training data")
    for e in range(1, 101):
        epoch_loss = train_models(
            embed_model, pred_model, train_data, device, lr_e=5e-4, lr_p=5e-4
        )
        print(f"{f'Epoch {e} loss:': <30} {epoch_loss}")

        if e % 5 == 0:
            val_acc = val_prediction(embed_model, pred_model, val_data, device)
            print(f"\033[1m{'Validation accuracy is:': <30} {val_acc}\033[0m")
    else:
        val_acc = val_prediction(embed_model, pred_model, val_data, device)
        print(f"\033[1m{'Validation accuracy is:': <30} {val_acc}\033[0m")
