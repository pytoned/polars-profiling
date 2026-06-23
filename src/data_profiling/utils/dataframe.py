"""Utils for pandas DataFrames."""
import hashlib
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Optional

import polars as pl


def warn_read(extension: str) -> None:
    """Warn the user when an extension is not supported.

    Args:
        extension: The extension that is not supported.
    """
    warnings.warn(
        f"""There was an attempt to read a file with extension {extension}, we assume it to be in CSV format.
To prevent this warning from showing up, please rename the file to any of the extensions supported by pandas
(docs: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)
If you think this extension should be supported, please report this as an issue:
https://github.com/Data-Centric-AI-Community/data-profiling/issues"""
    )


def is_supported_compression(file_extension: str) -> bool:
    """Determine if the given file extension indicates a compression format that pandas can handle automatically.

    Args:
        file_extension (str): the file extension to test

    Returns:
        bool: True if the extension indicates a compression format that pandas handles automatically and False otherwise

    Notes:
        Pandas can handle on the fly decompression from the following extensions: ‘.bz2’, ‘.gz’, ‘.zip’, or ‘.xz’
        (otherwise no decompression). If using ‘.zip’, the ZIP file must contain exactly one data file to be read in.
    """
    return file_extension.lower() in [".bz2", ".gz", ".xz", ".zip"]


def remove_suffix(text: str, suffix: str) -> str:
    """Removes the given suffix from the given string.

    Args:
        text (str): the string to remove the suffix from
        suffix (str): the suffix to remove from the string

    Returns:
        str: the string with the suffix removed, if the string ends with the suffix, otherwise the unmodified string

    Notes:
        In python 3.9+, there is a built-in string method called removesuffix() that can serve this purpose.
    """
    return text[: -len(suffix)] if suffix and text.endswith(suffix) else text


def uncompressed_extension(file_name: Path) -> str:
    """Returns the uncompressed extension of the given file name.

    Args:
        file_name (Path): the file name to get the uncompressed extension of

    Returns:
        str: the uncompressed extension, or the original extension if pandas doesn't handle it automatically
    """
    extension = file_name.suffix.lower()
    return (
        Path(remove_suffix(str(file_name).lower(), extension)).suffix
        if is_supported_compression(extension)
        else extension
    )


def read_pandas(file_name: Path) -> "pl.DataFrame":
    """Read a Polars DataFrame based on the file extension.

    Various file types are supported (.csv, .json, .jsonl, .tsv, .xls(x),
    .parquet, .ipc/.arrow/.feather).

    Args:
        file_name: the file to read

    Returns:
        A Polars DataFrame.

    Notes:
        This function is not intended to be flexible or complete. The main use
        case is to be able to read files without user input. For more advanced
        use cases, the user should load the DataFrame in code.
    """
    extension = uncompressed_extension(file_name)
    if extension == ".json":
        df = pl.read_json(str(file_name))
    elif extension == ".jsonl":
        df = pl.read_ndjson(str(file_name))
    elif extension == ".tsv":
        df = pl.read_csv(str(file_name), separator="\t")
    elif extension in [".xls", ".xlsx"]:
        df = pl.read_excel(str(file_name))
    elif extension == ".parquet":
        df = pl.read_parquet(str(file_name))
    elif extension in [".ipc", ".arrow", ".feather"]:
        df = pl.read_ipc(str(file_name))
    else:
        if extension != ".csv":
            warn_read(extension)
        df = pl.read_csv(str(file_name))
    return df


def rename_index(df: "pl.DataFrame") -> "pl.DataFrame":
    """Rename a reserved ``index`` column to ``df_index``.

    Args:
        df: DataFrame to process.

    Returns:
        The DataFrame with the ``index`` column renamed to ``df_index`` (if any).
    """
    if "index" in df.columns:
        df = df.rename({"index": "df_index"})
    return df


# Change this if `hash_dataframe`'s implementation changes.
HASH_PREFIX = "2@"


def hash_dataframe(df: "pl.DataFrame") -> str:
    """Hash a Polars DataFrame (implementation might change in the future).

    Args:
        df: the DataFrame

    Returns:
        The DataFrame's hash
    """
    hash_values = "\n".join(str(v) for v in df.hash_rows().to_list())
    digest = hashlib.sha256(hash_values.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}{digest}"


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def sort_column_names(dct: dict, sort: Optional[str]) -> dict:
    if sort is None:
        return dct

    sort = sort.lower()
    if sort.startswith("asc"):
        dct = dict(sorted(dct.items(), key=lambda x: x[0].casefold()))
    elif sort.startswith("desc"):
        dct = dict(sorted(dct.items(), key=lambda x: x[0].casefold(), reverse=True))
    else:
        raise ValueError('"sort" should be "ascending", "descending" or None.')
    return dct
