"""Character / word / length summaries for categorical & text variables.

These operate on the (small) value-counts container produced by the Polars
backend and return :class:`VarSeries` objects, so no pandas is involved.
The Unicode category/script/block logic is implemented off plain Python data
structures.
"""
import string
from collections import Counter
from typing import List

import numpy as np

from data_profiling.model.polars.utils_polars import weighted_median
from data_profiling.utils.varseries import VarSeries


def _counter_to_varseries(counter: Counter) -> VarSeries:
    if not counter:
        return VarSeries(np.array([], dtype=object), index=np.array([], dtype=object))
    items, counts = zip(*counter.most_common())
    return VarSeries(np.asarray(counts), index=np.asarray(items, dtype=object))


def length_summary_vc(vc: VarSeries) -> dict:
    length_counts: Counter = Counter()
    for category, freq in vc.items():
        length_counts[len(str(category))] += int(freq)

    if not length_counts:
        return {
            "max_length": np.nan,
            "mean_length": np.nan,
            "median_length": np.nan,
            "min_length": np.nan,
            "length_histogram": VarSeries(np.array([]), index=np.array([])),
        }

    # Sort descending by frequency (matches the original ordering).
    items = sorted(length_counts.items(), key=lambda kv: -kv[1])
    lengths = np.asarray([k for k, _ in items])
    counts = np.asarray([v for _, v in items])

    return {
        "max_length": int(np.max(lengths)),
        "mean_length": float(np.average(lengths, weights=counts)),
        "median_length": weighted_median(lengths, counts),
        "min_length": int(np.min(lengths)),
        "length_histogram": VarSeries(counts, index=lengths),
    }


def word_summary_vc(vc: VarSeries, stop_words: List[str] = []) -> dict:
    word_counts: Counter = Counter()
    strip_chars = string.punctuation + string.whitespace
    for category, freq in vc.items():
        for word in str(category).lower().split():
            word = word.strip(strip_chars)
            if word:
                word_counts[word] += int(freq)

    if stop_words:
        lowered = {w.lower() for w in stop_words}
        for w in list(word_counts.keys()):
            if w in lowered:
                del word_counts[w]

    if not word_counts:
        return {}
    return {"word_counts": _counter_to_varseries(word_counts)}


def _character_counts(vc: VarSeries) -> Counter:
    char_counts: Counter = Counter()
    for category, freq in vc.items():
        for char in str(category):
            char_counts[char] += int(freq)
    return char_counts


def unicode_summary_vc(vc: VarSeries) -> dict:
    try:
        from tangled_up_in_unicode import (  # type: ignore
            block,
            block_abbr,
            category,
            category_long,
            script,
        )
    except ImportError:
        from unicodedata import category as _category

        category = _category  # type: ignore
        _unknown = lambda char: "(unknown)"  # noqa: E731
        block = _unknown
        block_abbr = _unknown
        category_long = _unknown
        script = _unknown

    character_counts = _character_counts(vc)

    summary = {
        "n_characters_distinct": len(character_counts),
        "n_characters": int(sum(character_counts.values())),
        "character_counts": _counter_to_varseries(character_counts),
    }

    char_to_block = {key: block(key) for key in character_counts}
    char_to_category_short = {key: category(key) for key in character_counts}
    char_to_script = {key: script(key) for key in character_counts}

    summary["category_alias_values"] = {
        key: category_long(value) for key, value in char_to_category_short.items()
    }
    summary["block_alias_values"] = {
        key: block_abbr(value) for key, value in char_to_block.items()
    }

    # Per-block character distribution
    block_alias_counts: Counter = Counter()
    per_block_char_counts = {k: Counter() for k in summary["block_alias_values"].values()}
    for char, n_char in character_counts.items():
        block_name = summary["block_alias_values"][char]
        block_alias_counts[block_name] += n_char
        per_block_char_counts[block_name][char] = n_char
    summary["block_alias_counts"] = _counter_to_varseries(block_alias_counts)
    summary["n_block_alias"] = len(summary["block_alias_counts"])
    summary["block_alias_char_counts"] = {
        k: _counter_to_varseries(v) for k, v in per_block_char_counts.items()
    }

    # Per-script character distribution
    script_counts: Counter = Counter()
    per_script_char_counts = {k: Counter() for k in char_to_script.values()}
    for char, n_char in character_counts.items():
        script_name = char_to_script[char]
        script_counts[script_name] += n_char
        per_script_char_counts[script_name][char] = n_char
    summary["script_counts"] = _counter_to_varseries(script_counts)
    summary["n_scripts"] = len(summary["script_counts"])
    summary["script_char_counts"] = {
        k: _counter_to_varseries(v) for k, v in per_script_char_counts.items()
    }

    # Per-category character distribution
    category_alias_counts: Counter = Counter()
    per_category_alias_char_counts = {
        k: Counter() for k in summary["category_alias_values"].values()
    }
    for char, n_char in character_counts.items():
        category_alias_name = summary["category_alias_values"][char]
        category_alias_counts[category_alias_name] += n_char
        per_category_alias_char_counts[category_alias_name][char] += n_char
    # Replace underscores with spaces in the category names.
    category_alias_counts = Counter(
        {k.replace("_", " "): v for k, v in category_alias_counts.items()}
    )
    summary["category_alias_counts"] = _counter_to_varseries(category_alias_counts)
    summary["n_category"] = len(summary["category_alias_counts"])
    summary["category_alias_char_counts"] = {
        k: _counter_to_varseries(v) for k, v in per_category_alias_char_counts.items()
    }

    return summary
