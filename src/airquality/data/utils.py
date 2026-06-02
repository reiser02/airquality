import glob
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from airquality.config import cfg_get_str



class UnsupportedFileFormatError(Exception):
    """Exception raised when the file format is not supported."""
    pass


def ensure_datetime_series(series: pd.Series, *, freq: str, name: str) -> pd.Series:
    """Validate and normalize one pandas series to a regular datetime grid."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"La serie '{name}' debe ser pd.Series; recibido: {type(series)}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"La serie '{name}' debe tener DatetimeIndex")

    out = series.sort_index().copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.asfreq(freq)
    out.name = str(series.name) if series.name is not None else name
    return out.astype(float)


def load_dataset_paths(
    base_path: str = cfg_get_str("data", "base_path_glob", "../../data/*/"),
    key_word: str = cfg_get_str("data", "key_word", "CO_media_horaria"),
    file_extension: str = cfg_get_str("data", "file_extension", "json"),
) -> List[str]:
    """
    Search for files in the base_path whose names include key_word 
    and return a list of paths.

    Args:
        base_path: Full path to the files
        key_word: Used to match and identify the target files.
    """
    return glob.glob(os.path.join(base_path, f"*{key_word}*.{file_extension}"))


def load_to_df(file_path: str, name_from_path: bool = True) -> Optional[pd.DataFrame]:
    """
    Reads a JSON file, converts it to a DataFrame, and sets the hourly frequency.
    
    Args:
        file_path: Full path to the file.
        name_from_path: If True, gets column name from filename (split by '_').
                        If False, keeps the original column name from file.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    try:
        if extension == '.json':
            # Load data
            data_series = pd.read_json(file_path, typ='series')
            df = pd.DataFrame(data_series.rows, columns=data_series.cols)

            # Time transformations
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index(df.columns[0], inplace=True)
            df = df.asfreq('h')
        elif extension == '.csv':
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            raise UnsupportedFileFormatError(f"Unsupported file format: '{extension}'. Only .json and .csv are supported.")

        # Column naming logic
        if name_from_path:
            column_name = os.path.basename(file_path).split("_")[0]
            df.columns = [column_name]
        
        return df
    except UnsupportedFileFormatError as e:
        print(f"Skipping file {file_path}: {e}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    

def get_longest_segment(dfs: list[pd.DataFrame], force_end: bool = False, w_col: float = 0.6, w_row: float = 0.4, verbose: bool = False) -> pd.DataFrame:
    """
    Identifies and returns the optimal continuous segment of non-null values based on a weighted score or a mandatory end-point.

    Args:
        dfs (list[pd.DataFrame]): A list of DataFrames to be concatenated and analyzed.
        force_end (bool): If True, only considers the continuous block connected to 
            the most recent timestamp. If False, searches for the best segment Added weights to prioritize more columns or more rows in global search.
            anywhere in the timeline.
        w_col (float): Weight assigned to the number of columns (width) when 
            calculating the segment score. Defaults to 0.6.
        w_row (float): Weight assigned to the number of rows (length) when 
            calculating the segment score. Defaults to 0.4.
        verbose (bool): If True, prints a summary of the winning segment's 
            dimensions, recovered columns, and date range.

    Returns:
        pd.DataFrame: A DataFrame containing the selected continuous block of 
            valid data and its corresponding complete columns.
    """

    df_concat = (
        pd.concat(dfs, axis=1, join='outer')
        .sort_index()
        .asfreq('h')
    )

    if force_end:
        # Last timestamp
        last_time = df_concat.index.max()

        # Non-NaN columns at the last timestamp
        active_cols = df_concat.loc[last_time].dropna().index

        # Mask for valid rows
        valid_rows = df_concat[active_cols].notna().all(axis=1)

        # Continuous blocks
        blocks = (valid_rows != valid_rows.shift()).cumsum()

        # Identify the block containing the last timestamp
        last_block_id = blocks.loc[last_time]

        # Extract block
        best_df = df_concat.loc[blocks == last_block_id, active_cols].copy()

        if verbose:
            print(f"Number of matching series: {best_df.shape[1]}")
            print(f"Range: {best_df.index.min()} to {best_df.index.max()}")
            print(f"Total time points recovered: {len(best_df)}")
    else:
        # Pre-calculate data presence
        coincidence_count = df_concat.notna().sum(axis=1)
        max_possible_cols = int(coincidence_count.max())
        
        best_score = 0
        best_df = pd.DataFrame()

        # Search for the best compromise
        for n_cols in range(1, max_possible_cols + 1):
            # Filter rows that have at least 'n_cols'
            valid_mask = (coincidence_count >= n_cols)
            
            if not valid_mask.any():
                continue
                
            # Group continuous blocks
            blocks = (valid_mask != valid_mask.shift()).cumsum()
            valid_blocks = blocks[valid_mask]
            
            # Iterate through each continuous block found for this n_cols
            for block_id in valid_blocks.unique():
                df_block = df_concat.loc[blocks == block_id]
                
                # KEY IDENTIFICATION: 
                # Keep only columns that are 100% complete in this block
                complete_cols = df_block.columns[df_block.notna().all()]
                
                # If the number of complete columns is >= our current requirement
                actual_n_cols = len(complete_cols)
                actual_rows = len(df_block)
                score = (actual_n_cols ** w_col) * (actual_rows ** w_row)
                
                if score > best_score:
                    best_score = score
                    # Return only the columns that won the consensus
                    best_df = df_block[complete_cols].copy()

        if verbose and not best_df.empty:
            print(f"--- Winning Segment ---")
            print(f"Dimensions: {best_df.shape[0]} rows x {best_df.shape[1]} columns")
            print(f"Recovered columns: {list(best_df.columns)}")
            print(f"Range: {best_df.index.min()} to {best_df.index.max()}")
            print(f"Score (Total area): {best_score}")
            print(f"------------------------")
        
    return best_df
