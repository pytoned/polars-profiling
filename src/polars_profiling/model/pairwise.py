from typing import Any, List, Tuple

import polars as pl

from polars_profiling.config import Settings
from polars_profiling.visualisation.plot import scatter_pairwise


def get_scatter_tasks(
    config: Settings, continuous_variables: list
) -> List[Tuple[Any, Any]]:
    if not config.interactions.continuous:
        return []

    targets = config.interactions.targets
    if len(targets) == 0:
        targets = continuous_variables

    tasks = [(x, y) for y in continuous_variables for x in targets]
    return tasks


def get_scatter_plot(
    config: Settings, df: pl.DataFrame, x: Any, y: Any, continuous_variables: list
) -> str:
    if x in continuous_variables:
        if y == x:
            df_temp = df.select([pl.col(x)]).drop_nulls()
            xs = ys = df_temp.get_column(x).to_numpy()
        else:
            df_temp = df.select([pl.col(x), pl.col(y)]).drop_nulls()
            xs = df_temp.get_column(x).to_numpy()
            ys = df_temp.get_column(y).to_numpy()
        return scatter_pairwise(config, xs, ys, x, y)
    else:
        return ""
