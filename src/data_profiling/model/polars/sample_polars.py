"""Sampling for the Polars backend."""
from typing import List

import polars as pl

from data_profiling.config import Settings
from data_profiling.model.sample import Sample, get_sample


@get_sample.register(Settings, pl.DataFrame)
def polars_get_sample(config: Settings, df: pl.DataFrame) -> List[Sample]:
    """Obtain head/tail/random samples from a Polars DataFrame."""
    samples: List[Sample] = []
    if df.height == 0:
        return samples

    n_head = config.samples.head
    if n_head > 0:
        samples.append(Sample(id="head", data=df.head(n_head), name="First rows"))

    n_tail = config.samples.tail
    if n_tail > 0:
        samples.append(Sample(id="tail", data=df.tail(n_tail), name="Last rows"))

    n_random = config.samples.random
    if n_random > 0:
        samples.append(
            Sample(
                id="random",
                data=df.sample(n=n_random),
                name="Random sample",
            )
        )

    return samples
