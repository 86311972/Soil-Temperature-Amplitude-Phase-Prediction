# Soil Temperature Amplitude and Phase Prediction

MATLAB code for the paper:

**"Harmonic-Based Soil Temperature Simulation and Thermal Regime Mapping in Iran Using MODIS LST and the Sinusoidal Heat Transfer Equation"**

## Author
Hadi Zare (Khormizi)

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

## Usage
1. Place `train_final.xlsx` in the `data/` folder.
2. Open `STAmp_STPhase_Prediction.m` in MATLAB.
3. Select the target: `TS_Amp` or `TS_Phase`.
4. Run the script. Results will be saved in the `results/` folder.
