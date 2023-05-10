import re
from typing import Generator, Optional, Tuple

import polars as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset, random_split

import helper_funcs

WINNER_REGEX = "^winner_.+$"
BATCH_SIZE = 512

"""
Data creation for simple and embedding neural networks
"""


class PlayTransformer:
    def _preprocess(self, df, fit=False):
        for color in ["red", "blue"]:
            df = df.with_column(
                pl.col(f"{color}_general_runes")
                .apply(lambda p: p[0])
                .alias(f"{color}_keystone")
            )
        new_df = pl.get_dummies(
            df,
            columns=[
                "blue_champion",
                "red_champion",
                "blue_keystone",
                "red_keystone",
                "winner",
            ],
        )

        for l in [
            x for n, x in enumerate(df.columns) if df.dtypes[n] == pl.List(pl.Utf8)
        ]:
            new_df = new_df.hstack(
                df.select(pl.col(l))
                .with_row_count()
                .explode(l)
                .with_columns(pl.lit(1).alias("__one__"))
                .pivot(index="row_nr", columns=l, values="__one__")
                .select(pl.exclude(["row_nr", l]).prefix(f"{l}_"))
                .fill_null(0)
            )

        new_df = new_df.with_column(
            pl.col([pl.UInt8, pl.Int32, pl.Int64]).cast(pl.Float32)
        )

        # Tried to include above in the select(pl.exclude(['row_nr', l])) but didn't work
        new_df = new_df.select(pl.exclude(pl.List(pl.Utf8)))

        # Testing all features
        new_df = new_df.select(pl.exclude("^.+_.+_gold$"))

        level_cols = pl.col("^*_level$")
        gold_cols = pl.col("^*_gold$")

        if fit:
            self.max_gold_expr = new_df.select(
                pl.max(new_df.select(gold_cols)).max()
            ).item()
            self.min_gold_exp = new_df.select(
                pl.max(new_df.select(gold_cols)).min()
            ).item()
            self.max_level_expr = new_df.select(
                pl.max(new_df.select(level_cols)).max()
            ).item()
            self.min_level_exp = new_df.select(
                pl.max(new_df.select(level_cols)).min()
            ).item()
        if fit:
            self.string_cache = pl.StringCache()

        with self.string_cache:
            numeric_df = (
                new_df.lazy()
                .with_column(
                    gold_cols.map(
                        lambda s: (
                            (s - self.min_gold_exp)
                            / (self.max_gold_expr - self.min_gold_exp)
                        )
                    )
                )
                .with_column(
                    level_cols.map(
                        lambda s: (
                            (s - self.min_level_exp)
                            / (self.max_level_expr - self.min_level_exp)
                        )
                    )
                )
                .with_column(pl.col(pl.Utf8).cast(pl.Categorical))
                .with_column(pl.col(pl.Categorical).to_physical())
                .fill_null(-1)
                .collect()
            )

        numeric_df = numeric_df.select(pl.exclude("^.+_general_runes.+$"))

        if fit:
            self.all_columns = numeric_df.columns
        else:
            # Add 0 inplace of all columns missing in the inference df, then select them to make sure order is the name
            # if order isn't the same as training then when turning into numpy array everything may be misalinged
            numeric_df = numeric_df.with_columns(
                [
                    pl.lit(0).alias(col)
                    for col in (set(self.all_columns) - set(numeric_df.columns))
                ]
            ).select(self.all_columns)

        return numeric_df

    def fit(self, df):
        self._preprocess(df, fit=True)

    def transform(self, df):
        return self._preprocess(df, fit=False)

    def fit_transform(self, df):
        return self._preprocess(df, fit=True)


# TODO: Consider using lazy dataframe and scan_parquet for large data
def creat_dataframe(sample: Optional[int] = None) -> pl.DataFrame:
    df = pl.read_parquet(
        "kills_df_model_ready_v2.parquet",
        n_rows=sample,
    )

    play_transformer = PlayTransformer()
    numeric_df = play_transformer.fit_transform(df)

    return numeric_df


def seperate_data(d: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    return d.select(pl.exclude(WINNER_REGEX)), d.select(pl.col(WINNER_REGEX))


class LolData(Dataset):
    def __init__(self, data: pl.DataFrame) -> None:
        super().__init__()
        self.data, self.label = seperate_data(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data[idx].to_numpy()
        row_label = self.label[idx].to_numpy()
        return torch.tensor(row, dtype=torch.float32).squeeze(), torch.tensor(
            row_label, dtype=torch.float32
        )


def make_loader(*ds: Subset, batch_size: int = 32) -> Generator:
    return (
        DataLoader(
            d, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
        )
        for d in ds
    )


def get_dataloaders(sample_size: Optional[int]) -> Tuple[DataLoader, ...]:
    numeric_df = creat_dataframe()
    if sample_size:
        numeric_df = numeric_df.sample(sample_size)
    train_data, test_data = helper_funcs.split_polars(numeric_df, split=0.2)
    train_dataset = LolData(train_data)
    train_dataload, val_dataload = make_loader(
        *random_split(
            train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(42)
        ),
        batch_size=BATCH_SIZE,
    )

    return train_dataload, val_dataload
