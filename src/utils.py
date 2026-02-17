import glob
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

class UnsupportedFileFormatError(Exception):
    """Exception raised when the file format is not supported."""
    pass

def load_dataset_paths(base_path: str = "../../data/*/", key_word: str = "CO_media_horaria", file_extension: str = "json") -> List[str]:
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