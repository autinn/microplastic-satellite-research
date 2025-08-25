# Microplastic Concentration Forecasting

A comprehensive toolkit for forecasting microplastic concentration in ocean waters using statistical modeling, deep learning, and clustering approaches, with a focus on waters around Japan.

![error_dbscan_cnnlstm_SA](https://github.com/user-attachments/assets/137f4c96-4104-4788-8e51-20861d7e6d92)
![error_dbscan_convlstm](https://github.com/user-attachments/assets/e10a0f0a-17d3-4483-af31-508dbae811b5)

# Overview
This repository contains implementations and results from a multi-method analysis of spatiotemporal forecasting for microplastic concentration. The research explores:
- Clustering Analysis: DBSCAN for identifying high-concentration regions
- Regression Analysis: Multivariable regression analysis with 8 environmental variables
- Statistical Models: ARIMA, SARIMA, ARIMAX with Optuna hyperparameter tuning
- Deep Learning Approaches: CNN-LSTM and ConvLSTM architectures with Spatial Attention
- Transformer-based Models: Spacetimeformer for spatiotemporal prediction
- Advanced Data Processing: Reversible Instance Normalization (RevIN) for better handling of non-stationary data

# Exploratory Data Analysis
- Clustering analysis with DBSCAN
- Regression modeling with feature importance analysis

# Statistical Modeling
- ARIMA parameter tuning with grid search optimization
- SARIMA for seasonal patterns in microplastic distribution
- ARIMAX incorporating exogenous variables
- Optuna-based hyperparameter optimization

# Deep Learning Models
- CNN-LSTM: Basic, Spatial Attention, and RevIN variants
- ConvLSTM: Basic, Spatial Attention, and RevIN variants

# Transformer Models
- Spacetimeformer implementation for microplastic forecasting
- Attention-based mechanisms for capturing long-range dependencies
- Reference: Long-Range Transformers for Dynamic Spatiotemporal Forecasting (https://arxiv.org/pdf/2109.12218)

# Data Sources
The microplastic concentration data is derived from NASA's CYGNSS mission datasets:
1. Level 3 Ocean Microplastic Concentration Version 3.2: https://podaac.jpl.nasa.gov/dataset/CYGNSS_L3_MICROPLASTIC_V3.2
2. Level 2 Ocean Surface Heat Flux Science Data Record Version 3.2: https://podaac.jpl.nasa.gov/dataset/CYGNSS_L2_SURFACE_FLUX_V3.2
3. Level 3 Science Data Record Version 3.2: https://podaac.jpl.nasa.gov/dataset/CYGNSS_L3_V3.2 

# Key Results
- Spatial attention mechanisms significantly improve prediction accuracy in areas with high microplastic concentration variability
- RevIN normalization techniques boost model performance for non-stationary time series
- Transformer-based approaches capture long-range dependencies better than CNN-LSTM for longer forecast horizons
- DBSCAN clustering effectively identifies coherent high-concentration regions of microplastics


# Acknowledgments
This research was conducted at Minerva AI Sustainability Lab in collaboration with The Nippon Foundation from Japan.
The satellite data is obtained from NASA's CYNGSS project and is directly downloaded from the website: 

