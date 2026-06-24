"""
    Auxiliary handler methods for data summary extraction
"""
from typing import Any, Callable, Dict, List, Sequence


def compose(functions: Sequence[Callable]) -> Callable:
    """
    Compose a sequence of functions.

    :param functions: sequence of functions
    :return: combined function applying all functions in order.
    """

    def composed_function(*args) -> List[Any]:
        result = args  # Start with the input arguments
        for func in functions:
            result = func(*result) if isinstance(result, tuple) else func(result)
        return result  # type: ignore

    return composed_function  # type: ignore


class Handler:
    """A generic handler

    Allows any custom mapping between data types and functions
    """

    def __init__(
        self,
        mapping: Dict[str, List[Callable]],
        typeset: Any,
        *args,
        **kwargs
    ):
        self.mapping = mapping
        self.typeset = typeset
        self._complete_dag()

    def _complete_dag(self) -> None:
        """Prepend the base-type functions to every concrete type.

        Every variable type derives from the base ``Unsupported`` type, so the
        shared bookkeeping (counts, generic stats, supported stats) runs before
        the type-specific description.
        """
        base_type = getattr(self.typeset, "base_type", "Unsupported")
        base_funcs = self.mapping.get(base_type, [])
        for type_name in list(self.mapping.keys()):
            if type_name != base_type:
                self.mapping[type_name] = base_funcs + self.mapping[type_name]

    def handle(self, dtype: str, *args, **kwargs) -> dict:
        """
        Returns:
            object: a tuple containing the config, the dataset series and the summary extracted
        """
        funcs = self.mapping.get(dtype, [])
        op = compose(funcs)
        summary = op(*args)[-1]
        return summary


def get_render_map() -> Dict[str, Callable]:
    import polars_profiling.report.structure.variables as render_algorithms

    render_map = {
        "Boolean": render_algorithms.render_boolean,
        "Numeric": render_algorithms.render_real,
        "Complex": render_algorithms.render_complex,
        "Text": render_algorithms.render_text,
        "DateTime": render_algorithms.render_date,
        "Categorical": render_algorithms.render_categorical,
        "URL": render_algorithms.render_url,
        "Path": render_algorithms.render_path,
        "File": render_algorithms.render_file,
        "Image": render_algorithms.render_image,
        "Unsupported": render_algorithms.render_generic,
        "TimeSeries": render_algorithms.render_timeseries,
    }

    return render_map
