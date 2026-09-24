# Soil-Temperature-Amplitude-Phase-Prediction
MATLAB code for predicting soil temperature amplitude (STAmp) and phase (STPhase) using MLR, RF, and Ensemble models with 10-fold cross-validation.

## Models
- MLR-Full (Multiple Linear Regression with all predictors)
- MLR-Sig (Multiple Linear Regression with significant predictors)
- RF-Full (Random Forest with all predictors)
- Ensemble (MLR-Full + RF-Full)

## Methodology
- 10-fold cross-validation with 10 repetitions (100 assessments)
- Common fold assignment across all models for fair comparison

## Input Data
- 8 predictors: LST_Amp, LST_Phase, NDVI, Albedo, ST, BD, SOC, WV
- 319 meteorological stations (2022)

## Outputs
- Model performance metrics (R², RMSE, MAE)
- Variable importance
- Out-of-fold predictions

## Requirements
- MATLAB R2024b
- Statistics and Machine Learning Toolbox
