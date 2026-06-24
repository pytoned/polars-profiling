import polars as pl

from polars_profiling.report.presentation.core.duplicate import Duplicate
from polars_profiling.report.presentation.flavours.html import templates
from polars_profiling.report.presentation.flavours.html.frame import frame_to_html


def to_html(df: pl.DataFrame) -> str:
    html = frame_to_html(df, classes="duplicate table table-striped")
    if df.height == 0:
        html = html.replace(
            "<tbody>",
            f"<tbody><tr><td colspan={len(df.columns) + 1}>Dataset does not contain duplicate rows.</td></tr>",
        )
    return html


class HTMLDuplicate(Duplicate):
    def render(self) -> str:
        duplicate_html = to_html(self.content["duplicate"])
        return templates.template("duplicate.html").render(
            **self.content, duplicate_html=duplicate_html
        )
