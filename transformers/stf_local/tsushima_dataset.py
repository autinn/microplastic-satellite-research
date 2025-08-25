import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Any, Dict

# Global variables for map dimensions (will be set by the main script after loading preprocessed data)
TARGET_MAP_HEIGHT = None
TARGET_MAP_WIDTH = None

class TsushimaSpacetimeDataset(Dataset):
    """
    Custom PyTorch Dataset for Tsushima microplastic data,
    formatting it into Spacetimeformer's spatiotemporal token sequence.
    """
    def __init__(self, data: np.ndarray, timestamps: torch.Tensor,
                 look_back_window: int, prediction_horizon: int, 
                 num_variates: int):
        """
        Initializes the dataset.
        Args:
            data (np.ndarray): Normalized NumPy array of shape (time, height, width).
            timestamps (torch.Tensor): Time features for each timestep, shape (time, num_time_features).
            look_back_window (int): Number of past timesteps to use as context (L).
            prediction_horizon (int): Number of future timesteps to predict (H).
            num_variates (int): Total number of spatial variables (height * width).
        """
        self.data = data # (time, height, width)
        self.timestamps = timestamps # (time, num_time_features)
        self.look_back_window = look_back_window
        self.prediction_horizon = prediction_horizon
        self.num_variates = num_variates
        self.num_total_timesteps = self.data.shape[0]
        self.num_time_features = self.timestamps.shape[-1] # d_x for the model

        # Pre-calculate base IDs for efficiency. These are relative positions.
        self.base_time_ids_context = np.arange(self.look_back_window)
        self.base_time_ids_predict = np.arange(self.prediction_horizon) + self.look_back_window
        self.base_variate_ids = np.arange(self.num_variates)

    def __len__(self):
        """
        Returns the total number of samples (sequences) available in the dataset.
        Each sample requires 'look_back_window' past timesteps and 'prediction_horizon' future timesteps.
        """
        return self.num_total_timesteps - self.look_back_window - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Retrieves a single sample (context and target) from the dataset.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing (inputs_dict, target_tensor).
                   inputs_dict: Dictionary of encoder/decoder input tensors for Spacetimeformer.
                   target_tensor: Actual future values (ground truth) for prediction.
        """
        # Determine the start index for slicing the full data and timestamps
        start_data_idx = idx
        end_data_idx = idx + self.look_back_window + self.prediction_horizon

        # --- Slice data and timestamps for the full sequence (context + target) ---
        # data_slice: (look_back_window + prediction_horizon, height, width)
        data_slice = self.data[start_data_idx : end_data_idx]
        # timestamps_slice: (look_back_window + prediction_horizon, num_time_features)
        timestamps_slice = self.timestamps[start_data_idx : end_data_idx]

        # --- Flatten data_slice for Spacetimeformer 'values' input ---
        # Shape: (total_seq_len * num_variates, 1)
        full_values_flat = data_slice.reshape(-1, 1)

        # --- Prepare time_ids for the full sequence ---
        # These are relative time steps for positional encoding, repeated for each variate
        # Shape: (total_seq_len * num_variates,)
        full_time_ids = np.concatenate([
            np.repeat(self.base_time_ids_context, self.num_variates),
            np.repeat(self.base_time_ids_predict, self.num_variates)
        ])

        # --- Prepare variate_ids for the full sequence ---
        # These are variate IDs, tiled for each time step
        # Shape: (total_seq_len * num_variates,)
        full_variate_ids = np.tile(self.base_variate_ids, self.look_back_window + self.prediction_horizon)

        # --- Prepare explicit time features (x_mark_enc/dec) ---
        # These are the actual calendar features (year, month, day, hour), repeated for each variate
        # Shape: (total_seq_len * num_variates, num_time_features)
        full_x_mark = timestamps_slice.repeat_interleave(self.num_variates, dim=0)


        # --- Split into Encoder (Context) and Decoder (Target) parts ---
        enc_len = self.look_back_window * self.num_variates
        dec_len = self.prediction_horizon * self.num_variates

        # Encoder inputs
        x_enc_values = torch.from_numpy(full_values_flat[:enc_len]).float()
        x_enc_time_ids = torch.from_numpy(full_time_ids[:enc_len]).long()
        x_enc_variate_ids = torch.from_numpy(full_variate_ids[:enc_len]).long()
        x_mark_enc = full_x_mark[:enc_len].float() # Explicit time features for encoder

        # Decoder inputs
        # x_dec_values are zeros as we are predicting these
        x_dec_values = torch.zeros((dec_len, 1), dtype=torch.float32)
        x_dec_time_ids = torch.from_numpy(full_time_ids[enc_len:]).long()
        x_dec_variate_ids = torch.from_numpy(full_variate_ids[enc_len:]).long()
        x_mark_dec = full_x_mark[enc_len:].float() # Explicit time features for decoder

        # Target (ground truth for decoder output)
        y_target_values = torch.from_numpy(full_values_flat[enc_len:]).float()

        return {
            'x_enc_values': x_enc_values,
            'x_enc_time_ids': x_enc_time_ids, # Relative time IDs for positional embedding
            'x_enc_variate_ids': x_enc_variate_ids,
            'x_mark_enc': x_mark_enc, # Explicit calendar time features
            'x_dec_values': x_dec_values,
            'x_dec_time_ids': x_dec_time_ids, # Relative time IDs for positional embedding
            'x_dec_variate_ids': x_dec_variate_ids,
            'x_mark_dec': x_mark_dec # Explicit calendar time features
        }, y_target_values

