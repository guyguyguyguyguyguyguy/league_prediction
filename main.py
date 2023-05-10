import re
import statistics
from functools import partial
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import torch
import torch.nn.functional as F
import torch_geometric.nn
import torch_geometric.transforms
import torch_geometric.utils
from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.loader import NeighborLoader

from make_dataset import creat_dataframe, seperate_data

plt.style.use("grayscale")

"""
This is how data needs to look to use torch_geometric (taken from their website):

  The following describes a undirectged graph with three nodes and four edges. Each node has one feature:

    edge_index = torch.tensor([[0, 1, 1, 2],
                              [1, 0, 2, 1]], dtype=torch.long) -> edge index, to be read (0, 1), (1, 0), (1, 2), (2, 1)

    x = torch.tensor([[-1], [0], [1]], dtype=torch.float) -> Node features

    data = Data(x=x, edge_index=edge_index)
      Gives: Data(edge_index=[2, 4], x=[3, 1]) # 3 nodes, each with a feature. 4 edges each consisting of 2 nodes (guess you can have hypergraphs: one edge with n nodes)


    For undirected graphs, both direction of the edge should be added.

"""

r"""
How to make data?

    [=> is the current choice]

    NODE REPRESENTATION 
        1) For each player, create all possible nodes and only have edges betwen nodes that have a non-zero feature, eg. 1 if catagorical, float if money etc. (other nodes have no edges and a value of 0)

        2) For each player, only create nodes that have a value, with that node being assigned some distinct value (not sure how), edges between all nodes.

        => 3) Each node represents some subset of the columns, eg. one node for runes, one node for character. The features for each node are of length n (number of possible values for that feature), with a 1 denoting the player has that feature. Edges between all nodes. 


    GRAPH STRUCTURE
        => 1) Create a complete bipartite graph. Each player represents a disjoint set of nodes which has edges to all nodes of the other player. For each attribute of player A this represents the relationship said attribute and the whole of the player B's state. Then we integrate over all over player A's attributes so get a local and global attribute effect. This representes a relative embedding of a players state.
        For each node of player 1 => 
                p1      p2              p1 (no inter-node edges)
                /-------o              o
                o--------o              o
                \-------o              o
                \------o              o
                \-----o              o
                    ...                 ...
                    ...                 ...
                    ...                 ...
                        \o              0

        2) Matchup is represtned by two disjoint, complete graphs, represeting each player. All the attributes of a player are have edges between each other. No edges between players. This represents a more absolute embedding of a players state.

    PREDICTION MODEL 
        1) EDGE PREDICTION TASK: winner has an edge to the 'winner' node. The loser has no out-going edge

        2) CLASSIFICATION TASK: no winner node. Train network to classify which graph wins. Similar to paper, https://arxiv.org/pdf/2210.09517.pdf.

"""


class LolGraphData:
    """
    Class to transform the data format from the input into its herterogeneous bipartite graph form. Specifically the created graph has nodes of different types while all edges are of the same type.
    """

    def __init__(
        self,
        data: pl.DataFrame,
    ) -> None:
        super().__init__()
        player_data, winner_data = seperate_data(data)
        self.graph_data = self.create_graph(player_data, winner_data)

    @staticmethod
    def create_node_features(df: pl.DataFrame) -> Tuple[pl.DataFrame, ...]:

        columns = " ".join(df.columns)
        columns = list(set(re.findall(r"\s?([a-z]+_[a-z]+)", columns)))

        _df = (
            df.lazy()
            .with_columns(
                [(pl.concat_list(pl.col(f"^{x}.*$")).alias(x)) for x in columns]
            )
            .select(pl.col(pl.List(pl.Float32)))
            .collect()
        )

        p1_df = _df.select(pl.col(f"^blue.*$"))
        p2_df = _df.select(pl.col(f"^red.*$"))

        p1_df.columns = list(map(lambda x: x.removeprefix("blue_"), p1_df.columns))
        p2_df.columns = list(map(lambda x: x.removeprefix("red_"), p2_df.columns))

        return p1_df, p2_df

    def create_graph(self, p_df: pl.DataFrame, w_df: pl.DataFrame) -> HeteroData:

        p1, p2 = self.create_node_features(p_df)
        ps = p1.vstack(p2.select(p1.columns), in_place=False)
        ps_dict = ps.to_dict(as_series=False)
        ps_dict_t = {k: torch.tensor(v) for k, v in ps_dict.items()}
        half_size = ps.shape[0] // 2

        hetero_graph = HeteroData()
        for k, v in ps_dict_t.items():
            hetero_graph[k].x = v
        hetero_graph["player"].x = torch.ones(half_size, 1)  # type: ignore
        hetero_graph["player"].y = torch.tensor(w_df.to_numpy(), dtype=torch.float)  # type: ignore

        half_edges = torch.vstack(
            [torch.arange(0, half_size), torch.arange(half_size, half_size * 2)]
        )

        for d in ps.columns:
            for r in ps.columns:
                hetero_graph[d, r].edge_index = torch.cat(
                    [half_edges, half_edges.flip(dims=(0,))], dim=1
                )

            hetero_graph["player", d].edge_index = torch.arange(
                0, ps.shape[0] // 2
            ).repeat(2, 1)

            hetero_graph[d, "player"].edge_index = torch.arange(
                0, ps.shape[0] // 2
            ).repeat(2, 1)

        return hetero_graph


# This uses GraphSAGE: https://arxiv.org/pdf/1706.02216.pdf
class GNN(torch.nn.Module):
    """
    Graph neural network class, assumes all nodes and edges are of the same type
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels)
        self.conv2 = torch_geometric.nn.SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        if self.training:
            return x

        # TODO: Not working
        return F.softmax(x, dim=1)


class HNN(torch.nn.Module):
    """
    Heterogeneous graph neural network class, can take nodes and edges of different types
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        data: HeteroData,
    ) -> None:
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = torch_geometric.nn.Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = torch_geometric.nn.HGTConv(
                hidden_channels,
                hidden_channels,
                data.metadata(),
                num_heads,
                group="sum",
            )
            self.convs.append(conv)

        self.lin = torch_geometric.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict: Dict, edge_index_dict: Dict) -> torch.Tensor:
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        if self.training:
            return self.lin(x_dict["player"])
        return F.softmax(self.lin(x_dict["player"]), dim=1)


# https://arxiv.org/pdf/2205.09310.pdf
class _ECELoss(torch.nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t=1.0):
        softmaxes = F.softmax(logits / t, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


# https://arxiv.org/pdf/2205.09310.pdf
class LogitNormLoss(torch.nn.Module):
    def __init__(self, device: torch.device, t: float = 1.0):
        super().__init__()
        self.device = device
        self.t = t

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


class StubSmugLoss(torch.nn.Module):
    """
    Custom loss function to control the spread of probabilities produced. However, this loss is not validated to ensure correctness.
    """

    def __init__(
        self, device: torch.device, *, stub: float = 0.5, smug: float = 0.5
    ) -> None:
        """
        smug: float
            Punishes the model being too confidnet in correct guesses (in forward) and in guesses that predict winning (in __forward)
        stub: float
            Punishes the model being too confidnet in wrong guesses (in forward) and in guesses that predict losing (in __forward)
        """
        super().__init__()
        self.device = device
        self.stub_smug_p = torch.tensor([stub, smug]).to(device)

    # This provides differential penalisation for winning and losing guesses
    def other_forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smug_stub_pen = x * target.mul(self.stub_smug_p)
        return F.binary_cross_entropy_with_logits(x, target) + smug_stub_pen.mean()

    # This provides differential penalisation for correct and incorrect guesses
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        correct_wrong_x = x.gather(
            1, target.to(int)
        )  # puts all wrong gueses in the first column and correct guesses in second
        stub_smug_pen = torch.abs(
            correct_wrong_x * self.stub_smug_p
        )  # abs required as logits can be negative
        return F.binary_cross_entropy_with_logits(x, target) + stub_smug_pen.mean()


def train(
    model: HNN, train_loader: NeighborLoader, device: torch.device
) -> Tuple[float, ...]:
    model.train()
    model.to(device)

    # loss_fn = LogitNormLoss(device, 1)
    loss_fn = StubSmugLoss(device, stub=0.45, smug=0.12)
    # loss_fn = torch.nn.BCEWithLogitsLoss()

    total_examples = total_loss = 0
    accuracy = []
    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        batch_size = batch["player"].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = loss_fn(out, batch["player"].y)
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        accuracy.append(
            (out.argmax(axis=1) == batch["player"].y.argmax(axis=1))
            .to(torch.float32)
            .cpu()
            .mean()
        )

    return total_loss / total_examples, float(np.mean(accuracy))


@torch.no_grad()
def validate(
    model: HNN, val_graph: HeteroData, device: torch.device, cal: bool = False
) -> float | Tuple[float, torch.Tensor]:
    model.eval()
    model.to(device)

    val_graph.to(device)
    out = model(val_graph.x_dict, val_graph.edge_index_dict)
    acc = (
        (out.argmax(axis=1) == val_graph["player"].y.argmax(axis=1))
        .to(torch.float32)
        .cpu()
        .mean()
        .item()
    )

    if cal:
        return acc, _ECELoss()(out, val_graph["player"].y.argmax(axis=1))

    return acc


def graph_inference(
    model: HNN, matchup: pl.DataFrame, device: torch.device
) -> npt.NDArray:
    matchup_graph = LolGraphData(matchup)
    matchup_graph_n = torch_geometric.transforms.NormalizeFeatures()(
        matchup_graph.graph_data
    )
    matchup_graph_n.rename("items", "c_items")

    model.eval()
    model.to(device)

    matchup_graph_n.to(device)
    out = model(matchup_graph_n.x_dict, matchup_graph_n.edge_index_dict)

    return out.squeeze().detach().numpy()


def plot_confidence(
    m: torch.Tensor,
    ax: plt.Axes,
    title: Optional[str] = None,
    ylim: Optional[None] = None,
    max_v: float = 1.0,
) -> None:

    q = torch.linspace(0.5, max_v, 10)
    bs = torch.bucketize(m, q)

    idx, y = torch.unique(bs.detach().cpu(), return_counts=True)
    x = q[idx]

    # ax.plot(x, y, "-o", zorder=1)
    bars = ax.bar(x, height=y, width=(max_v - 0.5) / 11, alpha=0.5, zorder=0)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            round(bar.get_height(), 1),
            horizontalalignment="center",
            # weight="bold",
        )
    if ylim:
        ax.set_ylim(0, ylim)
    if title:
        ax.set_title(title)


def visualise_confidence(
    model: torch.nn.Module, val_graph: HeteroData, device: torch.device
) -> None:
    model.eval()
    model.to(device)
    val_graph.to(device)

    out = model(val_graph.x_dict, val_graph.edge_index_dict)
    label = val_graph["player"].y.argmax(axis=1)
    conf, pre_label = torch.max(out, dim=1)

    correct_mask = (label.eq(pre_label)).squeeze()
    winning = label.eq(1).squeeze()

    conf_correct_win = conf[correct_mask & winning]
    conf_correct_los = conf[correct_mask & ~winning]
    conf_wrong_win = conf[~correct_mask & winning]
    conf_wrong_los = conf[~correct_mask & ~winning]

    _, ax = plt.subplots(2, 2, figsize=(10, 16))
    ylim = conf_correct_win.size(0)
    max_val = conf.max().item()
    limmed_plot = partial(plot_confidence, max_v=max_val, ylim=ylim)
    limmed_plot(conf_correct_win, ax[0][0], "Correct winning predictions")
    limmed_plot(
        conf_wrong_win,
        ax[0][1],
        "Wrong winning predictions (flase positives)",
    )
    limmed_plot(conf_correct_los, ax[1][0], "Correct losing predictions")
    limmed_plot(
        conf_wrong_los,
        ax[1][1],
        "Wrong losing predictions (false negatives)",
    )
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_mathups = 50_000
    df = creat_dataframe(num_mathups)
    train_val_df, inf_df = df[:-1], df[-1]
    train_graph_data = LolGraphData(train_val_df[: int(num_mathups * 0.9)]).graph_data
    val_graph_data = LolGraphData(train_val_df[int(num_mathups * 0.9) :]).graph_data

    normal_graph = torch_geometric.transforms.NormalizeFeatures()
    train_graph_data = normal_graph(train_graph_data)
    val_graph_data = normal_graph(val_graph_data)

    train_graph_data.rename("items", "c_items")
    val_graph_data.rename("items", "c_items")

    loader = NeighborLoader(
        train_graph_data,
        [15] * 4,
        batch_size=512,
        input_nodes=("player", None),
    )

    # model = GNN(hidden_channels=64, out_channels=2)
    # model = torch_geometric.nn.to_hetero(
    #     model, train_graph_data.metadata(), aggr="mean"
    # )

    model = HNN(
        hidden_channels=64,
        out_channels=2,
        num_heads=2,
        num_layers=2,
        data=train_graph_data,
    )

    with torch.no_grad():
        model(train_graph_data.x_dict, train_graph_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    for e in range(1, 6):
        epoch_loss, epoch_acc = train(model, loader, device)
        print(f"{f'Epoch {e} loss:': <30} {epoch_loss}")
        print(f"{f'Epoch {e} accuracy is:': <30} {epoch_acc}")

        if e % 5 == 0:
            val_acc = validate(model, val_graph_data, device)
            print(f"\033[1m{f'Validation accuracy is:': <30} {val_acc}\033[0m")

    # calibration testing
    else:
        val_acc, val_cal = validate(model, val_graph_data, device, cal=bool)
        print(f"\033[1m{f'Validation accuracy is:': <30} {val_acc}\033[0m")
        print(f"\033[1m{f'Validation calibration is:': <30} {val_cal.item()}\033[0m")

    # torch.save(model.state_dict(), "HNN_model_state_dict.pt")
    inf_prediction = graph_inference(model, inf_df, device)
    result = "Win" if inf_df["winner_blue"].item() else "Loss"
    print(
        f"\nPrediction: \n \t win: {inf_prediction[0]}, loss: {inf_prediction[1]} \n Actual: {result}"
    )
    visualise_confidence(model, val_graph_data, device)
