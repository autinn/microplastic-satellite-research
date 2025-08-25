import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional
import torch

# --- Configuration ---
DATA_DIR = './data/' # Directory containing your raw .nc files
TARGET_VARIABLE = 'mp_concentration'

# Tsushima Bounding Box

TSUSHIMA_LAT_MIN, TSUSHIMA_LON_MIN = 32.97713,125.788
TSUSHIMA_LAT_MAX, TSUSHIMA_LON_MAX = 35.3734,133.611

# Paths for saving preprocessed data and scaler
PREPROCESSED_DATA_PATH = 'tsushima_normalized_microplastic_data.nc'
SCALER_PATH = 'tsushima_minmax_scaler.joblib'

# This ratio must be consistent with the TRAIN_SPLIT_RATIO in your main training script
TRAIN_SPLIT_RATIO = 0.8 

# Limit the number of .nc files to process for preprocessing
MAX_FILES_TO_PROCESS = 500 

# --- Preprocessing Function ---
def preprocess_tsushima_data(data_dir: str, target_variable: str, 
                             lat_min: float, lat_max: float, lon_min: float, lon_max: float, 
                             preprocessed_path: str, scaler_path: str, 
                             train_split_ratio: float, max_files: Optional[int] = None):
    """
    Loads raw NetCDF data, crops to bounding box, handles NaNs, normalizes (fitting only on train split),
    and saves the processed data and the fitted scaler.
    max_files: If specified, limits the number of .nc files to process.
    """
    if os.path.exists(preprocessed_path) and os.path.exists(scaler_path):
        print(f"Preprocessed data and scaler already exist. Skipping preprocessing.")
        print(f"Loaded from: {preprocessed_path} and {scaler_path}")
        return

    print("Starting data preprocessing for Tsushima region...")
    
    try:
        # Get list of all .nc files and sort them
        nc_files_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')])
        
        # Limit the number of files if max_files is specified
        if max_files is not None and len(nc_files_list) > max_files:
            print(f"Limiting processing to the first {max_files} files out of {len(nc_files_list)}.")
            nc_files_list = nc_files_list[:max_files]

        # Load all .nc files into a single xarray Dataset
        ds = xr.open_mfdataset(nc_files_list, combine='by_coords', decode_times=True)
        data_array = ds[target_variable]
        data_array = data_array.sortby('time') # Ensure temporal order

        print(f"Original full dataset shape: {data_array.shape}")

        cropped_data_array = data_array.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
        if cropped_data_array.size == 0:
            raise ValueError(f"Cropping resulted in empty data. Check bounding box {lat_min}-{lat_max}, {lon_min}-{lon_max} against data coordinates.")

        print(f"Cropped data shape (Tsushima region): {cropped_data_array.shape}")

        np_data = cropped_data_array.values # Shape: (time, cropped_lat, cropped_lon)
        original_shape = np_data.shape # Store original shape before reshaping for scaler

        if np.isnan(np_data).any():
            print("Warning: NaN values found in cropped data. Filling with 0.0.")
            np_data = np.nan_to_num(np_data, nan=0.0)

        # --- IMPORTANT: Fit scaler ONLY on the training portion of the data ---
        num_total_timesteps = np_data.shape[0]
        scaler_train_split_idx = int(num_total_timesteps * train_split_ratio)

        scaler = MinMaxScaler(feature_range=(0, 1))
        # Reshape training data for scaler: (num_elements, 1)
        scaler.fit(np_data[:scaler_train_split_idx].reshape(-1, 1))

        # Transform the entire dataset using the scaler fitted on training data
        normalized_data = scaler.transform(np_data.reshape(-1, 1)).reshape(original_shape)
        
        normalized_data_array = xr.DataArray(
            normalized_data,
            coords=cropped_data_array.coords,
            dims=cropped_data_array.dims,
            name=f"normalized_{target_variable}"
        )

        normalized_data_array.to_netcdf(preprocessed_path)
        joblib.dump(scaler, scaler_path)
        print(f"Preprocessed Tsushima data saved to {preprocessed_path} and scaler to {scaler_path}")

        full_timestamps_dt = cropped_data_array['time'].dt
        timestamps_tensor = torch.stack([
            torch.tensor(full_timestamps_dt.year.values, dtype=torch.float32),
            torch.tensor(full_timestamps_dt.month.values, dtype=torch.float32),
            torch.tensor(full_timestamps_dt.day.values, dtype=torch.float32),
            torch.tensor(full_timestamps_dt.hour.values, dtype=torch.float32)
        ], dim=-1) # Shape: (num_total_timesteps, 4)
        
    except FileNotFoundError:
        print(f"Error: No .nc files found in {data_dir}. Please ensure the 'data' folder exists and contains .nc files.")
        print("Cannot proceed without raw data. Exiting.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        exit()
    
    
    return normalized_data_array, scaler, timestamps_tensor

# --- Main execution block for preprocessing ---
if __name__ == "__main__":
    print("=" * 60)
    print("TSUSHIMA MICROPLASTIC DATA PREPROCESSING SCRIPT")
    print("=" * 60)
    
    preprocess_tsushima_data(
        DATA_DIR, TARGET_VARIABLE,
        TSUSHIMA_LAT_MIN, TSUSHIMA_LAT_MAX, TSUSHIMA_LON_MIN, TSUSHIMA_LON_MAX,
        PREPROCESSED_DATA_PATH, SCALER_PATH,
        TRAIN_SPLIT_RATIO,
        max_files=MAX_FILES_TO_PROCESS # Pass the max_files parameter here
    )
    
    # Verify the saved data and scaler can be loaded
    try:
        loaded_data = xr.open_dataarray(PREPROCESSED_DATA_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)
        print(f"\nVerification: Successfully loaded preprocessed data (shape: {loaded_data.shape}) and scaler.")
        print(f"Inferred Tsushima map dimensions: {loaded_data.shape[1]}x{loaded_data.shape[2]}")
        print(f"Number of variates (N): {loaded_data.shape[1] * loaded_data.shape[2]}")
    except Exception as e:
        print(f"\nVerification failed: {e}")

    print("=" * 60)
    print("PREPROCESSING SCRIPT FINISHED.")
    print("=" * 60)
