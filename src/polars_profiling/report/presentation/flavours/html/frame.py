"""Render a Polars DataFrame to an HTML table (pandas ``to_html`` compatible)."""
import html as _html
from typing import Optional, Sequence

import polars as pl


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return _html.escape(str(value))


def frame_to_html(
    df: pl.DataFrame,
    classes: str = "",
    index_values: Optional[Sequence] = None,
) -> str:
    """Render ``df`` as an HTML table styled like ``pandas.DataFrame.to_html``."""
    columns = df.columns
    n = df.height
    index = list(index_values) if index_values is not None else list(range(n))

    cls = f"dataframe {classes}".strip()
    header = "".join(f"<th>{_html.escape(str(c))}</th>" for c in columns)

    body_rows = []
    for i, row in enumerate(df.iter_rows()):
        cells = "".join(f"<td>{_fmt(v)}</td>" for v in row)
        idx = index[i] if i < len(index) else i
        body_rows.append(f"<tr><th>{_html.escape(str(idx))}</th>{cells}</tr>")

    return (
        f'<table border="1" class="{cls}">'
        f'<thead><tr style="text-align: right;"><th></th>{header}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )
