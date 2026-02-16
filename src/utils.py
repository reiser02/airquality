import glob
import os
from typing import List, Optional

import pandas as pd

def load_dataset_paths(base_path: str = "../data/*/", key_word: str = "CO_media_horaria") -> List[str]:
    """
    Search for JSON files in the base_path whose names include key_word 
    and return a list of paths.

    Args:
        base_path: Full path to the JSON files
        key_word: Used to match and identify the target JSON files.
    """
    return glob.glob(os.path.join(base_path, f"*{key_word}*.json"))


def load_json_to_df(file_path: str, name_from_path: bool = True) -> Optional[pd.DataFrame]:
    """
    Reads a JSON file, converts it to a DataFrame, and sets the hourly frequency.
    
    Args:
        file_path: Full path to the JSON file.
        name_from_path: If True, gets column name from filename (split by '_').
                        If False, keeps the original column name from JSON.
    """
    try:
        # Load data
        data_series = pd.read_json(file_path, typ='series')
        df = pd.DataFrame(data_series.rows, columns=data_series.cols)

        # Time transformations
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index(df.columns[0], inplace=True)
        df = df.asfreq('h')

        # Column naming logic
        if name_from_path:
            column_name = os.path.basename(file_path).split("_")[0]
            df.columns = [column_name]
        
        return df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None