# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""NLP collection -- text label binarization functions.

Provides unsupervised and supervised multi-label binarization for text columns.
Converts comma/semicolon-separated label strings into long-format occurrence
tables with a ``value`` column, using automatic language detection,
lemmatization, and fuzzy matching.
"""
import pandas as pd
import re
from collections import Counter
from typing import List


def nlp_binarize_labels_auto(df: pd.DataFrame, column: str, max_labels: int = 30) -> pd.DataFrame:
    """Unsupervised multi-label binarizer.

    Pipeline: language detection → delimiter detection → lemmatization →
    accent removal → binary matrix generation → top-N filtering.

    Args:
        df: Input DataFrame.
        column: Column containing delimited label strings
            (e.g. ``"pain, headache, nausea"``).
        max_labels: Maximum number of label columns to keep (most frequent).

    Returns:
        Long-format DataFrame with the original columns where *column* now
        contains one detected label per row, plus a ``value`` column (0 or 1).
    """
    from choregraph._extras import optional_dep
    with optional_dep():
        from unidecode import unidecode
        import simplemma
        from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    series = df[column]
    # --- 1. LANGUAGE DETECTION ---
    # Sample some text for language detection
    sample_text = " ".join(series.dropna().astype(str).head(50))
    if not sample_text.strip():
        # Fallback to English if no text is found
        detected_lang = 'en'
    else:
        try:
            detected_lang = detect(sample_text)
        except Exception:
            # Fallback to English if detection fails
            detected_lang = 'en'
    # --- 2. DELIMITER DETECTION ---
    all_raw_text = "".join(series.dropna().astype(str).head(1000))
    delimiters = re.findall(r'[,;|]', all_raw_text)
    sep = Counter(delimiters).most_common(1)[0][0] if delimiters else None
    # --- 3. EXTRACTION AND LEMMATIZATION ---
    if sep:
        raw_tokens = series.dropna().astype(str).str.split(rf'\s*\{sep}\s*').explode()
    else:
        raw_tokens = series.dropna().astype(str)
    unique_raw_tokens = sorted(list(set(raw_tokens.str.strip())))
    mapping = {}
    for original in unique_raw_tokens:
        if not original:
            continue
        # Lowercase and strip
        word_for_lemma = original.lower().strip()
        # Lemmatize (e.g., shoulders -> shoulder)
        try:
            lemma = simplemma.lemmatize(word_for_lemma, lang=detected_lang)
        except Exception:
            # If language not supported by simplemma, use the word as is
            lemma = word_for_lemma
        # Remove accents and clean
        final_label = unidecode(lemma).replace(',', '.')
        mapping[original] = final_label
    # --- 4. RECONSTRUCTION ---
    def apply_mapping(text):
        if pd.isna(text) or str(text).lower() == 'nan' or not str(text).strip():
            return []
        tokens = [t.strip() for t in (str(text).split(sep) if sep else [text])]
        return list({mapping.get(t, t) for t in tokens if t})
    clean_lists = series.apply(apply_mapping)
    # --- 5. BINARY MATRIX GENERATION ---
    exploded = clean_lists.explode()
    if exploded.empty:
        return pd.DataFrame(index=df.index)
    matrix = pd.get_dummies(exploded).groupby(level=0).sum()
    # Ensure it aligns with original index (in case of empty or dropped rows)
    matrix = matrix.reindex(df.index).fillna(0).astype(int)
    # --- 6. LIMIT COLUMNS TO TOP N ---
    if len(matrix.columns) > max_labels:
        top_cols = matrix.sum().nlargest(max_labels).index
        matrix = matrix[top_cols]
    # --- 7. LONGIFY ---
    df_base = df.drop(columns=[column])
    df_wide = pd.concat([df_base, matrix], axis=1)
    label_cols = list(matrix.columns)
    id_vars = [c for c in df_wide.columns if c not in label_cols]
    df_long = df_wide.melt(id_vars=id_vars, value_vars=label_cols,
                           var_name=column, value_name="value")
    df_long["value"] = df_long["value"].fillna(0).astype(int)
    return df_long


def nlp_binarize_labels_hinted(df: pd.DataFrame, column: str, hints: List[str], max_labels: int = 30) -> pd.DataFrame:
    """Supervised multi-label binarizer using hint labels.

    Uses fuzzy string matching (Levenshtein distance, score cutoff 80) to map
    extracted tokens to the provided hint list, correcting typos and variations.

    Args:
        df: Input DataFrame.
        column: Column containing delimited label strings.
        hints: List of canonical label names to match against.
        max_labels: Maximum number of label columns to keep.

    Returns:
        Long-format DataFrame with the original columns where *column* now
        contains one matched hint label per row, plus a ``value`` column (0 or 1).
    """
    from choregraph._extras import optional_dep
    with optional_dep():
        from unidecode import unidecode
        from rapidfuzz import process

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if not hints:
        return pd.DataFrame(index=df.index)
    series = df[column]
    # 1. DELIMITER DETECTION
    # Basic cleaning for delimiter detection
    clean_sample = series.dropna().astype(str).head(1000).apply(lambda x: unidecode(x).lower().strip())
    all_raw_text = "".join(clean_sample.values)
    delimiters = re.findall(r'[,;|]', all_raw_text)
    sep = Counter(delimiters).most_common(1)[0][0] if delimiters else None
    # 2. TOKEN EXTRACTION
    if sep:
        raw_tokens = series.dropna().astype(str).str.split(rf'\s*\{sep}\s*').explode()
    else:
        raw_tokens = series.dropna().astype(str)
    unique_tokens = sorted(list(set(raw_tokens.str.strip())))
    # 3. FUZZY MATCHING TO HINTS
    mapping = {}
    clean_hints = [unidecode(h).lower().strip() for h in hints]
    for token in unique_tokens:
        if not token:
            continue
        token_clean = unidecode(token).lower().strip()
        # Levenshtein distance matching (score_cutoff=80)
        match = process.extractOne(token_clean, clean_hints, score_cutoff=80)
        if match:
            # match[2] is the index of the detected hint
            mapping[token] = hints[match[2]]
        else:
            # If no match, keep the token (unidecoded and cleaned)
            mapping[token] = token_clean
    # 4. RECONSTRUCTION
    def apply_mapping(text):
        if pd.isna(text) or str(text).lower() == 'nan' or not str(text).strip():
            return []
        tokens = [t.strip() for t in (str(text).split(sep) if sep else [text])]
        return list({mapping.get(t, unidecode(t).lower().strip()) for t in tokens if t})
    clean_lists = series.apply(apply_mapping)
    # 5. BINARY MATRIX GENERATION
    exploded = clean_lists.explode()
    if exploded.empty:
        return pd.DataFrame(index=df.index)
    matrix = pd.get_dummies(exploded).groupby(level=0).sum()
    # Ensure it aligns with original index
    matrix = matrix.reindex(df.index).fillna(0).astype(int)
    # 6. LIMIT COLUMNS TO TOP N
    if len(matrix.columns) > max_labels:
        top_cols = matrix.sum().nlargest(max_labels).index
        matrix = matrix[top_cols]
    # --- 7. LONGIFY ---
    df_base = df.drop(columns=[column])
    df_wide = pd.concat([df_base, matrix], axis=1)
    label_cols = list(matrix.columns)
    id_vars = [c for c in df_wide.columns if c not in label_cols]
    df_long = df_wide.melt(id_vars=id_vars, value_vars=label_cols,
                           var_name=column, value_name="value")
    df_long["value"] = df_long["value"].fillna(0).astype(int)
    return df_long
