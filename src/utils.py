import glob
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import warnings
import logging
from sktime.forecasting.fbprophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor
from sktime.forecasting.compose import make_reduction

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
    

def apply_symmetric_gaps(df, num_gaps, size_k):
    """
    Applies symmetric gaps (NaN values) to a DataFrame at random non-overlapping positions.
    """
    df_noisy = df.copy()
    n = len(df_noisy)
    
    # Define margins and distances
    # margin: distance to the dataset boundaries
    # min_separation: distance between the end of one gap and the start of the next
    margin = size_k + 20
    min_separation = size_k + 5 
    
    possible_indices = list(range(margin, n - margin - size_k))
    start_points = []
    
    # Selection of points without overlapping
    # Attempt to find points until reaching num_gaps or running out of options
    attempts = 0
    while len(start_points) < num_gaps and attempts < 1000:
        if not possible_indices:
            break
            
        new_start = np.random.choice(possible_indices)
        start_points.append(new_start)
        
        # REMOVE nearby indices to avoid overlap
        # Delete all indices that would cause the next gap 
        # to be too close to the one just created
        forbidden_zone = range(new_start - min_separation, new_start + min_separation)
        possible_indices = [i for i in possible_indices if i not in forbidden_zone]
        attempts += 1

    # Apply the gaps
    for start in start_points:
        df_noisy.iloc[start : start + size_k, :] = np.nan
        
    return df_noisy


def impute_prophet(series):
    forecaster = Prophet(
        add_country_holidays={"country_name": "Spain"},
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        verbose=False
    )
    
    # Train with known data (removing NaNs)
    # y_train maintains the original index
    y_train = series.dropna()
    
    if len(y_train) < 2: # Avoid errors if there are too many gaps
        return series
    
    logging.getLogger('prophet').setLevel(logging.ERROR)
    forecaster.fit(y_train)
    
    # Identify missing points (NaNs)
    missing_points = series[series.isna()].index
    
    if not missing_points.empty:
        # Predict for those points
        y_pred = forecaster.predict(fh=missing_points)
        
        # Combine: original values + predictions in the gaps
        imputed_series = series.copy()
        imputed_series.update(y_pred)
        return imputed_series
    
    return series


def impute_iterative(df):
    # Save indices and columns
    cols = df.columns
    idx = df.index
    
    # Normalize
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols, index=idx)
    
    # Impute
    imputer = IterativeImputer(max_iter=10, random_state=42)
    df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=cols, index=idx)
    
    # Denormalize
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=cols, index=idx)
    
    return df_imputed


def impute_lgbm(series, window_length=10):
    y_train = series.dropna()
    
    if len(y_train) < (window_length * 2):
        return series
    
    # Index reconstruction for sktime
    # Create a simple integer index (0, 1, 2...) for training.
    # This avoids issues with hourly frequency and NaNs,
    # allowing LightGBM to learn only from the value sequence.
    y_train_idx = y_train.copy()
    y_train_idx.index = range(len(y_train))
    
    try:
        regressor = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=616,
            verbosity=-1,
            n_jobs=1
        )
        
        # We use strategy="recursive". Sktime will treat the integer index
        # as time steps (t, t+1, t+2...)
        forecaster = make_reduction(regressor, window_length=window_length, strategy="recursive")
        forecaster.fit(y_train_idx)
        
        # Identify missing points and their locations
        missing_points = series[series.isna()].index
        
        if not missing_points.empty:
            # To predict, we need to tell sktime how many steps into the future to look.
            # fh (forecasting horizon) relative to the end of y_train_idx
            fh = list(range(1, len(missing_points) + 1))
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                y_pred = forecaster.predict(fh=fh)
            
            # Reassign predicted values to original indices (dates)
            imputed_series = series.copy()
            # Convert y_pred back to actual date indices
            final_predictions = pd.Series(y_pred.values, index=missing_points)
            
            imputed_series.update(final_predictions)
            return imputed_series
            
    except Exception:
        return series
        
    return series