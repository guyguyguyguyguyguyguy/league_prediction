import itertools
import random
import statistics

# from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data.dataloader import DataLoader

from make_dataset import get_dataloaders
from model_class import NewClassifier

NO_FEATURES = 789  # type: int
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(m: NewClassifier, data: DataLoader, epochs: int = 5) -> None:
    m.to(DEVICE)
    for _ in range(epochs):
        m.train()
        for ds, labs in data:
            ds = ds.to(DEVICE)
            labs = labs.to(DEVICE)
            preds = m(ds)
            labs = einops.rearrange(labs, "b h w -> b (h w)")
            m.optimisation_step(preds, labs)


def test_model(m: NewClassifier, test_data: DataLoader) -> float:

    m.to(DEVICE)
    m.eval()
    with torch.no_grad():
        test_acc = []
        for ds, labs in test_data:
            ds = ds.to(DEVICE)
            v_lab = labs.to(DEVICE)
            v_pred = m(ds)
            v_lab = einops.rearrange(v_lab, "b h w -> b (h w)")
            test_acc.append(
                (v_pred.argmax(axis=1) == v_lab.argmax(axis=1))
                .to(torch.float32)
                .cpu()
                .mean()
                .item()
            )
    return statistics.mean(test_acc)


class Island:
    best_fitness = 0

    def __init__(
        self, no_models: int, training_data: DataLoader, test_data: DataLoader
    ) -> None:
        self.no_models = no_models
        self.train_data = train_dataloader
        self.test_data = test_data
        self.models = self.generate_init_models(training_data)

    def generate_init_model_params(
        self,
    ) -> Iterable[Tuple[npt.NDArray, float, npt.NDArray]]:
        hidden_layers = [
            np.sort(np.random.randint(1, 50, n) * 10)
            for n in np.random.randint(1, 8, self.no_models)
        ]
        dropout = np.random.random(size=self.no_models)

        feature_indices = [
            np.random.choice(
                np.arange(NO_FEATURES), size=(np.random.randint(5, NO_FEATURES))
            )
            for _ in range(self.no_models)
        ]

        return zip(hidden_layers, dropout, feature_indices)

    def _generate_single_model(self, elem, train_data):
        hl, d, feature_indices = elem
        model = NewClassifier(
            NO_FEATURES, list(hl), feature_indices=feature_indices, dropout=d
        )
        train_model(model, train_data)
        return model

    def generate_init_models(self, train_data: DataLoader) -> List[NewClassifier]:
        generate_params = self.generate_init_model_params()
        # with Pool(processes=3) as p:
        #     models = p.map(self._generate_single_model,
        #                    [list(param) + [train_data] for param in generate_params])
        models = []
        for param in generate_params:
            models.append(self._generate_single_model(param, train_data))
        return models

    def generation(self) -> npt.NDArray:
        fitness = []
        for m in self.models:
            fitty = test_model(m, self.test_data)
            fitness.append(fitty)

        self.best_fitness = (
            max(fitness) if max(fitness) > self.best_fitness else self.best_fitness
        )
        return np.array(fitness)


class Overlord:

    """
    Evolutionary algorithm class to find the best meta-parameters for neural network models.
    """

    current_generation: int = 0
    fitess_progretion: List = []

    def __init__(
        self,
        train_data: DataLoader,
        test_data: DataLoader,
        no_models: Optional[List[int]] = None,
        mutation_rate: float = 0.2,
        migration_rate: float = 0.2,
    ) -> None:
        self.no_models = no_models if no_models else [10] * 3
        self.mutation_rate = mutation_rate
        self.migration_rate = migration_rate
        self.islands = self.initialise_islands(no_models, train_data, test_data)

    def initialise_islands(
        self, no_models: List[int], train_data: DataLoader, test_data: DataLoader
    ) -> List[Island]:
        islands = []
        for i, n in enumerate(no_models):
            island = Island(n, train_data, test_data)
            islands.append(island)

        return islands

    # Tournement selection, keep half
    # Can also keep best each generation if we want (currently not done)
    @staticmethod
    def survival(fitnesses: npt.NDArray, island: Island) -> None:
        best_model = np.argmax(fitnesses)
        new_models = [island.models[best_model]]
        starting_model_nums = len(fitnesses)
        chance_for_selection = np.ones_like(fitnesses, dtype=float)
        chance_for_selection[best_model] = 0.0

        for _ in range(starting_model_nums // 2):
            indvs = np.random.choice(
                np.arange(len(fitnesses)),
                size=starting_model_nums // 3,
                p=chance_for_selection / chance_for_selection.sum(),
            )
            tournement = fitnesses[indvs]
            winner = indvs[np.argmax(tournement)]
            chance_for_selection[winner] = 0.0
            new_models.append(island.models[winner])

        island.models = new_models

    @staticmethod
    def _reproduction(mates: List[NewClassifier]) -> NewClassifier:
        hidden = random.choice(mates).hidden
        dropout = random.choice(mates).dropout
        mates = sorted(mates, key=lambda x: len(x.feature_indices))
        cross_over_idx = random.randint(0, len(mates[0].feature_indices))
        feature_indices = np.hstack(
            [
                mates[0].feature_indices[:cross_over_idx],
                mates[1].feature_indices[cross_over_idx:],
            ]
        )

        return NewClassifier(
            NO_FEATURES, hidden, feature_indices=feature_indices, dropout=dropout
        )

    def reproduction(self, island: Island) -> None:
        mates = np.random.choice(
            island.models, size=(len(island.models), 2), replace=True
        )
        for m1, m2 in mates:
            new_m = self._reproduction([m1, m2])
            island.models.append(new_m)

    @staticmethod
    def _mutation(m: NewClassifier) -> NewClassifier:
        hidden = np.maximum(m.hidden + np.random.randint(-10, 10, len(m.hidden)), 1)
        dropout = (
            random.random() * 0.1
            if m.dropout is None
            else max(0, min((random.random() - 0.5) + m.dropout, 1))
        )
        toggled_feature_indices = np.random.choice(
            np.arange(NO_FEATURES), np.random.randint(0, 10)
        )
        feature_indices = np.setxor1d(m.feature_indices, toggled_feature_indices)

        return NewClassifier(
            NO_FEATURES, hidden, feature_indices=feature_indices, dropout=dropout
        )

    def mutation(self, island: Island) -> None:
        mutating_models = np.random.choice(
            island.models, size=int(len(island.models) * self.mutation_rate)
        )
        for m in mutating_models:
            new_m = self._mutation(m)
            island.models.append(new_m)

    def migration(self, island: Island) -> None:
        receiving_island = random.choice(self.islands)
        departing_model = random.choice(np.arange(len(island.models)))
        receiving_island.models.append(island.models.pop(departing_model))

    def generational_step(self, island: Island) -> float:
        island_fitnesses = island.generation()
        self.survival(island_fitnesses, island)
        self.reproduction(island)
        self.mutation(island)
        if random.random() <= self.migration_rate:
            self.migration(island)

        new_models = island.models[island.no_models // 2 :]
        for m in new_models:
            train_model(m, island.train_data)

        print(f"Best fitness: {island.best_fitness}\n")

        return island.best_fitness

    # TODO: Do parallelisation for gpu
    def parrallel_generational_steps(self, num_gens: int) -> None:

        if DEVICE == torch.device("cpu"):
            ctx = torch.multiprocessing.get_context(
                "spawn"
            )  # This does not share the parent's local variables between each thread, but instead instantiates new variables (and pyhton interpreters) for each thread. Therefore these is no sharing of the data or anything. Less efficient but works with minial modifications to code
            with ctx.Pool(processes=3) as p:
                for i in range(num_gens):
                    print(f"Generation {i}:")
                    best_fitneses = p.map(self.generational_step, self.islands)

                    self.fitess_progretion.append(best_fitneses)

        elif DEVICE == torch.device("cuda"):
            ...

    def run(self, num_generations: int = 100) -> None:
        self.parrallel_generational_steps(num_generations)


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_dataloaders(10000)

    over = Overlord(train_dataloader, val_dataloader, no_models=[4, 4, 4])

    num_generations = 3
    over.run(3)

    progpresh = over.fitess_progretion

    i1, i2, i3 = zip(*progpresh)
    x = np.linspace(0, 1, num_generations)
    # il, i2, i3 = np.split(progpresh, 3, axis=1)

    plt.scatter(x, i1)
    plt.scatter(x, i2)
    plt.scatter(x, i3)
    plt.show()
