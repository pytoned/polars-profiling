from polars_profiling.report.presentation.core.sample import Sample
from polars_profiling.report.presentation.flavours.html import templates
from polars_profiling.report.presentation.flavours.html.frame import frame_to_html


class HTMLSample(Sample):
    def render(self) -> str:
        sample_html = frame_to_html(
            self.content["sample"], classes="sample table table-striped"
        )
        return templates.template("sample.html").render(
            **self.content, sample_html=sample_html
        )
