import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process


def normalize_text(text: str) -> str:
    """
    Normalize text by:
    - Converting to lowercase
    - Removing accents
    - Standardizing apostrophes
    - Removing hyphens (so "saint-denis" matches "saintdenis")
    - Removing extra whitespace
    """
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("-", "")
    text = " ".join(text.split())
    return text


def discover_common_prefixes(
    texts: list[str],
    min_frequency: int = 2,
    min_length: int = 3,
    max_prefix_length: int = 50,
) -> list[str]:
    """
    Dynamically discover common prefixes in a collection of texts.

    Parameters
    ----------
    texts : list of str
        List of texts to analyze for common prefixes.
    min_frequency : int, default=2
        Minimum number of texts that must share a prefix.
    min_length : int, default=3
        Minimum length of prefixes to consider.
    max_prefix_length : int, default=50
        Maximum length of prefixes to consider.

    Returns
    -------
    list of str
        Common prefixes sorted by length (longest first).
    """
    if not texts:
        return []

    prefix_counts: dict[str, int] = {}
    for text in texts:
        text_lower = text.lower().strip()
        for length in range(
            min_length, min(max_prefix_length + 1, len(text_lower) + 1)
        ):
            prefix = text_lower[:length]
            if prefix.endswith(" ") or prefix.endswith("'"):
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    common_prefixes = [p for p, c in prefix_counts.items() if c >= min_frequency]

    # Sort longest-first so truncate_common_prefixes matches the most
    # specific prefix before falling back to a shorter one.
    # We intentionally keep *all* qualifying prefixes: a longer prefix like
    # "tj de la " does NOT subsume a shorter "tj de " because the shorter
    # one matches many texts that the longer one does not.
    common_prefixes.sort(key=len, reverse=True)
    return common_prefixes


def truncate_common_prefixes(
    text: str,
    common_prefixes: list[str] | None = None,
) -> str:
    """
    Remove common court prefixes to isolate the city/location name.

    Parameters
    ----------
    text : str
        Court name to process.
    common_prefixes : list of str, optional
        Prefixes to strip.  Falls back to hardcoded court prefixes when *None*.

    Returns
    -------
    str
        Text with the first matching prefix removed.
    """
    text_lower = text.lower().strip()

    if common_prefixes is None:
        common_prefixes = [
            "tribunal superieur d'appel de ",
            "tribunal superieur d'appel d'",
            "cour d'appel d'",
            "cour d'appel de ",
            "tribunal judiciaire de ",
            "tribunal judiciaire d'",
            "tribunal judiciaire des ",
            "tribunal judiciaire du ",
            "tj de ",
            "tj d'",
            "tj des ",
            "tj du ",
        ]

    for prefix in common_prefixes:
        if text_lower.startswith(prefix):
            return text_lower[len(prefix) :].strip()

    return text_lower


def create_location_mapping(
    list1: list[str],
    list2: list[str],
    threshold: int = 80,
    method: str = "token_sort_ratio",
    auto_detect_prefixes: bool = True,
    min_prefix_frequency: int = 2,
) -> pd.DataFrame:
    """
    Create a mapping between two sets of geographic locations using fuzzy matching.

    Parameters
    ----------
    list1 : list of str
        Locations to map *from* (appear in ``left_o`` column).
    list2 : list of str
        Locations to map *to* (appear in ``right_o`` column).
    threshold : int, default=80
        Minimum similarity score (0-100) to consider a match.
    method : str, default='token_sort_ratio'
        Fuzzy matching method: 'ratio', 'partial_ratio', 'token_sort_ratio',
        'token_set_ratio'.
    auto_detect_prefixes : bool, default=True
        Automatically discover common prefixes from data before matching.
    min_prefix_frequency : int, default=2
        Minimum frequency for prefix detection (used when *auto_detect_prefixes*
        is True).

    Returns
    -------
    pd.DataFrame
        Columns: ``left_o``, ``right_o``, ``left_normalized``,
        ``right_normalized``, ``similarity_score``.
    """
    # Discover or use default prefixes
    common_prefixes = None
    if auto_detect_prefixes:
        all_texts = list(list1) + list(list2)
        common_prefixes = discover_common_prefixes(
            all_texts,
            min_frequency=min_prefix_frequency,
            min_length=3,
            max_prefix_length=50,
        )
        print(f"Discovered {len(common_prefixes)} common prefixes:")
        for prefix in common_prefixes[:10]:
            print(f"  '{prefix}'")
        if len(common_prefixes) > 10:
            print(f"  ... and {len(common_prefixes) - 10} more")

    # Normalize and strip prefixes
    normalized1 = {
        loc: truncate_common_prefixes(normalize_text(loc), common_prefixes)
        for loc in list1
    }
    normalized2 = {
        loc: truncate_common_prefixes(normalize_text(loc), common_prefixes)
        for loc in list2
    }

    # Choose scorer
    scorer_map = {
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
    }
    scorer = scorer_map.get(method, fuzz.token_sort_ratio)

    # Build reverse lookup: normalized value -> original key(s)
    norm2_reverse: dict[str, str] = {v: k for k, v in normalized2.items()}

    mappings: list[dict] = []
    for loc1_orig, loc1_norm in normalized1.items():
        result = process.extractOne(loc1_norm, normalized2.values(), scorer=scorer)
        if result is None:
            continue

        matched_norm, score = result[0], result[1]

        # For very short normalized names, try partial matching as fallback
        if len(loc1_norm) <= 10 and score < 70:
            partial_result = process.extractOne(
                loc1_norm, normalized2.values(), scorer=fuzz.partial_ratio
            )
            if partial_result and partial_result[1] > score:
                matched_norm, score = partial_result[0], partial_result[1]

        if score > threshold:
            loc2_orig = norm2_reverse[matched_norm]
            mappings.append(
                {
                    "left_o": loc1_orig,
                    "right_o": loc2_orig,
                    "left_normalized": loc1_norm,
                    "right_normalized": matched_norm,
                    "similarity_score": round(score, 2),
                }
            )

    df = pd.DataFrame(mappings)
    if not df.empty:
        df = df.sort_values("similarity_score", ascending=False).reset_index(drop=True)
    return df
