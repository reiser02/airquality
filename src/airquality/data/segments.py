"""Utilities for selecting complete contiguous multiseries segments."""

from __future__ import annotations

import pandas as pd


def contiguous_observed_segments(series: pd.Series, min_len: int = 1) -> list[pd.Series]:
    """Split ``series`` into contiguous NaN-free runs of at least ``min_len`` points.

    Used to avoid gluing non-contiguous stretches with ``dropna()`` before
    seasonal decompositions or windowed detectors: at every gap boundary the
    daily phase would jump (e.g. 08:00 stitched to 17:00 of another day).
    """
    observed = series.notna()
    block_ids = (observed != observed.shift()).cumsum()
    return [
        series.loc[block.index]
        for _, block in observed.groupby(block_ids)
        if bool(block.iloc[0]) and len(block) >= int(min_len)
    ]


def get_longest_segment(
    dfs: list[pd.DataFrame],
    w_col: float = 0.6,
    w_row: float = 0.4,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return the best continuous block of complete observations."""
    df_concat = pd.concat(dfs, axis=1, join="outer").sort_index().asfreq("h")

    coincidence_count = df_concat.notna().sum(axis=1)
    max_possible_cols = int(coincidence_count.max())

    best_score = 0.0
    best_df = pd.DataFrame()

    for n_cols in range(1, max_possible_cols + 1):
        valid_mask = coincidence_count >= n_cols
        if not valid_mask.any():
            continue

        blocks = (valid_mask != valid_mask.shift()).cumsum()
        valid_blocks = blocks[valid_mask]

        for block_id in valid_blocks.unique():
            df_block = df_concat.loc[blocks == block_id]
            complete_cols = df_block.columns[df_block.notna().all()]
            actual_n_cols = len(complete_cols)
            actual_rows = len(df_block)
            score = (actual_n_cols**w_col) * (actual_rows**w_row)

            if score > best_score:
                best_score = score
                best_df = df_block[complete_cols].copy()

    if verbose and not best_df.empty:
        print("--- Winning Segment ---")
        print(f"Dimensions: {best_df.shape[0]} rows x {best_df.shape[1]} columns")
        print(f"Recovered columns: {list(best_df.columns)}")
        print(f"Range: {best_df.index.min()} to {best_df.index.max()}")
        print(f"Score (Total area): {best_score}")
        print("------------------------")

    return best_df
