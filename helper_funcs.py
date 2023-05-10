import math
from typing import Generator, Iterable, List, Tuple

import numpy as np
import polars as pl


def split_polars(df: pl.DataFrame, *, split: float) -> Tuple[pl.DataFrame, ...]:
    df = df.sample(frac=1)

    s = df.shape[0]
    split_point = int(s * split)

    return (
        df[split_point:],
        df[:split_point],
    )
