clc;
clear;
close all;

%% ========================================================================
%             SOIL TEMPERATURE AMPLITUDE AND PHASE PREDICTION
%       MLR-FULL | MLR-SIG | RF-FULL | ENSEMBLE
%
%  Companion code for the paper:
%  "Harmonic-Based Soil Temperature Simulation and Thermal Regime Mapping
%   in Iran Using Remote Sensing Data and the Sinusoidal Heat Transfer Equation"
%
%  Author : Hadi Zare (Khormizi)
%  Year   : 2026
% ========================================================================

% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('       SOIL TEMPERATURE AMPLITUDE AND PHASE PREDICTION\n');
fprintf('====================================================================\n');

%% ========================== USER SETTINGS ==============================

% ---------------- Input / Output ----------------
% By default, the script reads 'train_final.xlsx' from the current folder
% and saves all outputs to a subfolder named 'results'.

input_dir  = fullfile(pwd, 'data');
output_dir = fullfile(pwd, 'results');

if ~exist(input_dir, 'dir')
    error(['Input folder not found: ' input_dir ...
           '. Please create a "data" folder and place train_final.xlsx inside it.']);
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

filename = fullfile(input_dir, 'train_final.xlsx');

% ---------------- Target ----------------
% Choose one of the two targets:
%   'TS_Amp'   -> soil temperature amplitude  (STAmp)
%   'TS_Phase' -> soil temperature phase      (STPhase)
% The script must be run separately for each target.

target = 'TS_Amp';
% target = 'TS_Phase';

% ---------------- Predictors ----------------
predictors = {
    'LST_Amp'
    'LST_Phase'
    'NDVI'
    'Albedo'
    'STI'
    'BD'
    'SOC'
    'WV'
    };

% ---------------- Cross-Validation ----------------
numFolds = 10;   % number of folds
numReps  = 10;   % number of repetitions

% ---------------- Random Forest ----------------
numTrees    = 200;   % number of trees
minLeafSize = 25;    % minimum leaf size

% ---------------- Ensemble ----------------
MLR_weight = 0.50;
RF_weight  = 0.50;

% ---------------- Significance level ----------------
alpha = 0.05;

% ---------------- Random seed ----------------
rng(42);

%% ========================================================================
% 1. READ DATA
% ========================================================================

fprintf('\nReading data...\n');

data = readtable(filename);

fprintf('\nAvailable variables:\n');
disp(data.Properties.VariableNames');

%% ========================================================================
% 2. CHECK VARIABLES
% ========================================================================

all_vars = data.Properties.VariableNames;

required_vars = [predictors; {target}];

missing_vars = {};

for i = 1:length(required_vars)
    if ~ismember(required_vars{i}, all_vars)
        missing_vars{end+1} = required_vars{i}; %#ok<SAGROW>
    end
end

if ~isempty(missing_vars)
    fprintf('\nERROR: The following variables are missing:\n');
    for i = 1:length(missing_vars)
        fprintf('   %s\n', missing_vars{i});
    end
    error('Required variables are missing from the Excel file.');
end

%% ========================================================================
% 3. EXTRACT DATA
% ========================================================================

X = data{:, predictors};
y = data{:, target};

% Station names (optional)
if ismember('Station', data.Properties.VariableNames)
    station_names = data.Station;
else
    station_names = repmat({''}, height(data), 1);
end

if isstring(station_names)
    station_names = cellstr(station_names);
elseif iscategorical(station_names)
    station_names = cellstr(station_names);
end

%% ========================================================================
% 4. REMOVE MISSING VALUES
% ========================================================================

valid_rows = ~any(isnan(X), 2) & ~isnan(y);

X = X(valid_rows, :);
y = y(valid_rows);
station_names = station_names(valid_rows);

fprintf('\n');
fprintf('====================================================================\n');
fprintf('Target              : %s\n', target);
fprintf('Number of predictors: %d\n', length(predictors));
fprintf('Valid samples       : %d\n', size(X, 1));
fprintf('Folds               : %d\n', numFolds);
fprintf('Repetitions         : %d\n', numReps);
fprintf('Total CV folds      : %d\n', numFolds * numReps);
fprintf('====================================================================\n');

%% ========================================================================
% 5. GENERATE COMMON CROSS-VALIDATION FOLDS
% ========================================================================

fprintf('\nGenerating common cross-validation folds...\n');

n = size(X, 1);

fold_assignment = zeros(n, numReps);

for rep = 1:numReps
    rng(42 + rep);
    cv = cvpartition(n, 'KFold', numFolds);
    for fold = 1:numFolds
        test_idx = test(cv, fold);
        fold_assignment(test_idx, rep) = fold;
    end
end

fprintf('Common CV folds generated successfully.\n');

%% ========================================================================
% 6. MLR-FULL SIGNIFICANCE ANALYSIS
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('               MLR-FULL SIGNIFICANCE ANALYSIS\n');
fprintf('====================================================================\n');

tblMLR = array2table(X, 'VariableNames', predictors);
tblMLR.(target) = y;

formula = sprintf('%s ~ %s', target, strjoin(predictors, ' + '));

fprintf('\nMLR formula:\n');
fprintf('%s\n', formula);

mdl_MLR_Full_All = fitlm(tblMLR, formula);

%% ========================================================================
% 7. EXTRACT MLR-FULL COEFFICIENTS
% ========================================================================

coef_full      = mdl_MLR_Full_All.Coefficients;
coef_names_full = coef_full.Properties.RowNames;

Estimate_full = coef_full.Estimate;
SE_full       = coef_full.SE;
tStat_full    = coef_full.tStat;
pValue_full   = coef_full.pValue;

% Standardized coefficients
X_z = zscore(X);
y_z = zscore(y);

tblStd = array2table(X_z, 'VariableNames', predictors);
tblStd.(target) = y_z;

formula_std = sprintf('%s ~ %s', target, strjoin(predictors, ' + '));
mdl_std_full = fitlm(tblStd, formula_std);

coef_std_full = mdl_std_full.Coefficients.Estimate;
Beta_full = coef_std_full(2:end);   % remove intercept

%% ========================================================================
% 8. MLR-FULL VARIABLE IMPORTANCE
% ========================================================================

abs_beta_full = abs(Beta_full);

if sum(abs_beta_full) > 0
    Importance_full = 100 * abs_beta_full ./ sum(abs_beta_full);
else
    Importance_full = zeros(size(abs_beta_full));
end

%% ========================================================================
% 9. DETERMINE SIGNIFICANT PREDICTORS
% ========================================================================

p_predictors_full = pValue_full(2:end);
significant_mask  = p_predictors_full < alpha;

significant_predictors     = predictors(significant_mask);
non_significant_predictors = predictors(~significant_mask);

fprintf('\nMLR-FULL significance results:\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('%-15s %-15s %-15s\n', 'Predictor', 'p-value', 'Status');
fprintf('--------------------------------------------------------------------\n');

for i = 1:length(predictors)
    if p_predictors_full(i) < alpha
        status = 'SIGNIFICANT';
    else
        status = 'NOT SIGNIFICANT';
    end
    fprintf('%-15s %-15.6f %-15s\n', predictors{i}, p_predictors_full(i), status);
end

fprintf('--------------------------------------------------------------------\n');
fprintf('\nSignificant predictors:\n');
if isempty(significant_predictors)
    fprintf('None\n');
else
    for i = 1:length(significant_predictors)
        fprintf('   %s\n', significant_predictors{i});
    end
end

%% ========================================================================
% 10. MLR-FULL COEFFICIENT TABLE
% ========================================================================

MLR_Full_Coefficients = table();
MLR_Full_Coefficients.Predictor           = coef_names_full;
MLR_Full_Coefficients.Coefficient_B       = Estimate_full;
MLR_Full_Coefficients.SE                  = SE_full;
MLR_Full_Coefficients.tStatistic          = tStat_full;
MLR_Full_Coefficients.pValue              = pValue_full;
MLR_Full_Coefficients.Standardized_Beta   = [NaN; Beta_full];
MLR_Full_Coefficients.Importance_Percent  = [NaN; Importance_full];

%% ========================================================================
% 11. MLR-SIG MODEL
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('                    MLR-SIG MODEL\n');
fprintf('====================================================================\n');

if isempty(significant_predictors)

    fprintf('\nWARNING: No significant predictors found at p < %.2f.\n', alpha);

    X_sig  = zeros(n, 0);
    tblSig = table();
    tblSig.(target) = y;
    formula_sig = sprintf('%s ~ 1', target);
    mdl_MLR_Sig_All = fitlm(tblSig, formula_sig);

else

    sig_columns = ismember(predictors, significant_predictors);
    X_sig = X(:, sig_columns);

    tblSig = array2table(X_sig, 'VariableNames', significant_predictors);
    tblSig.(target) = y;

    formula_sig = sprintf('%s ~ %s', target, strjoin(significant_predictors, ' + '));
    fprintf('\nMLR-SIG formula:\n');
    fprintf('%s\n', formula_sig);

    mdl_MLR_Sig_All = fitlm(tblSig, formula_sig);

end

%% ========================================================================
% 12. MLR-SIG COEFFICIENTS
% ========================================================================

coef_sig        = mdl_MLR_Sig_All.Coefficients;
coef_names_sig  = coef_sig.Properties.RowNames;

Estimate_sig = coef_sig.Estimate;
SE_sig       = coef_sig.SE;
tStat_sig    = coef_sig.tStat;
pValue_sig   = coef_sig.pValue;

if ~isempty(significant_predictors)
    X_sig_z = zscore(X_sig);
    tblSigStd = array2table(X_sig_z, 'VariableNames', significant_predictors);
    tblSigStd.(target) = y_z;
    formula_sig_std = sprintf('%s ~ %s', target, strjoin(significant_predictors, ' + '));
    mdl_std_sig = fitlm(tblSigStd, formula_sig_std);
    coef_std_sig = mdl_std_sig.Coefficients.Estimate;
    Beta_sig = coef_std_sig(2:end);
else
    Beta_sig = [];
end

%% ========================================================================
% 13. MLR-SIG IMPORTANCE
% ========================================================================

if ~isempty(Beta_sig)
    abs_beta_sig = abs(Beta_sig);
    if sum(abs_beta_sig) > 0
        Importance_sig = 100 * abs_beta_sig ./ sum(abs_beta_sig);
    else
        Importance_sig = zeros(size(abs_beta_sig));
    end
else
    Importance_sig = [];
end

Beta_sig_all       = [NaN; Beta_sig];
Importance_sig_all = [NaN; Importance_sig];

MLR_Sig_Coefficients = table();
MLR_Sig_Coefficients.Predictor          = coef_names_sig;
MLR_Sig_Coefficients.Coefficient_B      = Estimate_sig;
MLR_Sig_Coefficients.SE                 = SE_sig;
MLR_Sig_Coefficients.tStatistic         = tStat_sig;
MLR_Sig_Coefficients.pValue             = pValue_sig;
MLR_Sig_Coefficients.Standardized_Beta  = Beta_sig_all;
MLR_Sig_Coefficients.Importance_Percent = Importance_sig_all;

%% ========================================================================
% 14. RANDOM FOREST SETTINGS
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('                     RANDOM FOREST\n');
fprintf('====================================================================\n');

fprintf('\nRandom Forest settings:\n');
fprintf('Trees        : %d\n', numTrees);
fprintf('Min leaf size: %d\n', minLeafSize);
fprintf('Predictors   : ALL (%d)\n', length(predictors));

%% ========================================================================
% 15. PREALLOCATE CV RESULTS
% ========================================================================

totalFolds = numReps * numFolds;

MLRFull_R2_Train   = zeros(totalFolds, 1);
MLRFull_R2_Test    = zeros(totalFolds, 1);
MLRFull_RMSE_Train = zeros(totalFolds, 1);
MLRFull_RMSE_Test  = zeros(totalFolds, 1);
MLRFull_MAE_Train  = zeros(totalFolds, 1);
MLRFull_MAE_Test   = zeros(totalFolds, 1);

MLRSig_R2_Train    = zeros(totalFolds, 1);
MLRSig_R2_Test     = zeros(totalFolds, 1);
MLRSig_RMSE_Train  = zeros(totalFolds, 1);
MLRSig_RMSE_Test   = zeros(totalFolds, 1);
MLRSig_MAE_Train   = zeros(totalFolds, 1);
MLRSig_MAE_Test    = zeros(totalFolds, 1);

RF_R2_Train        = zeros(totalFolds, 1);
RF_R2_Test         = zeros(totalFolds, 1);
RF_RMSE_Train      = zeros(totalFolds, 1);
RF_RMSE_Test       = zeros(totalFolds, 1);
RF_MAE_Train       = zeros(totalFolds, 1);
RF_MAE_Test        = zeros(totalFolds, 1);

ENS_R2_Train       = zeros(totalFolds, 1);
ENS_R2_Test        = zeros(totalFolds, 1);
ENS_RMSE_Train     = zeros(totalFolds, 1);
ENS_RMSE_Test      = zeros(totalFolds, 1);
ENS_MAE_Train      = zeros(totalFolds, 1);
ENS_MAE_Test       = zeros(totalFolds, 1);

MLRFull_OOF = NaN(n, 1);
MLRSig_OOF  = NaN(n, 1);
RF_OOF      = NaN(n, 1);
ENS_OOF     = NaN(n, 1);

FoldNumber = zeros(totalFolds, 1);
RepNumber  = zeros(totalFolds, 1);

%% ========================================================================
% 16. CROSS-VALIDATION
% ========================================================================

fprintf('\nStarting %d-fold × %d-repetition cross-validation...\n', ...
    numFolds, numReps);

idx = 0;

for rep = 1:numReps

    fprintf('\nRepetition %d / %d\n', rep, numReps);

    for fold = 1:numFolds

        idx = idx + 1;
        fprintf('   Fold %2d / %2d\n', fold, numFolds);

        test_idx  = fold_assignment(:, rep) == fold;
        train_idx = ~test_idx;

        FoldNumber(idx) = fold;
        RepNumber(idx)  = rep;

        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test  = X(test_idx, :);
        y_test  = y(test_idx);

        % ---------------- MLR-FULL ----------------
        tbl_train_full = array2table(X_train, 'VariableNames', predictors);
        tbl_train_full.(target) = y_train;

        formula_full = sprintf('%s ~ %s', target, strjoin(predictors, ' + '));
        mdl = fitlm(tbl_train_full, formula_full);

        y_train_pred_MLR = predict(mdl, tbl_train_full);

        tbl_test_full = array2table(X_test, 'VariableNames', predictors);
        tbl_test_full.(target) = y_test;
        y_test_pred_MLR = predict(mdl, tbl_test_full);

        [r2tr, rmsetr, maetr] = calculate_metrics(y_train, y_train_pred_MLR);
        [r2te, rmsete, maete] = calculate_metrics(y_test,  y_test_pred_MLR);

        MLRFull_R2_Train(idx)   = r2tr;
        MLRFull_R2_Test(idx)    = r2te;
        MLRFull_RMSE_Train(idx) = rmsetr;
        MLRFull_RMSE_Test(idx)  = rmsete;
        MLRFull_MAE_Train(idx)  = maetr;
        MLRFull_MAE_Test(idx)   = maete;
        MLRFull_OOF(test_idx)   = y_test_pred_MLR;

        % ---------------- MLR-SIG ----------------
        if isempty(significant_predictors)
            y_train_pred_SIG = repmat(mean(y_train), length(y_train), 1);
            y_test_pred_SIG  = repmat(mean(y_train), length(y_test),  1);
        else
            sig_cols = ismember(predictors, significant_predictors);
            Xtr_sig = X_train(:, sig_cols);
            Xte_sig = X_test(:,  sig_cols);

            tbl_tr_sig = array2table(Xtr_sig, 'VariableNames', significant_predictors);
            tbl_tr_sig.(target) = y_train;

            formula_sig_cv = sprintf('%s ~ %s', target, strjoin(significant_predictors, ' + '));
            mdl_sig = fitlm(tbl_tr_sig, formula_sig_cv);
            y_train_pred_SIG = predict(mdl_sig, tbl_tr_sig);

            tbl_te_sig = array2table(Xte_sig, 'VariableNames', significant_predictors);
            tbl_te_sig.(target) = y_test;
            y_test_pred_SIG = predict(mdl_sig, tbl_te_sig);
        end

        [r2tr, rmsetr, maetr] = calculate_metrics(y_train, y_train_pred_SIG);
        [r2te, rmsete, maete] = calculate_metrics(y_test,  y_test_pred_SIG);

        MLRSig_R2_Train(idx)   = r2tr;
        MLRSig_R2_Test(idx)    = r2te;
        MLRSig_RMSE_Train(idx) = rmsetr;
        MLRSig_RMSE_Test(idx)  = rmsete;
        MLRSig_MAE_Train(idx)  = maetr;
        MLRSig_MAE_Test(idx)   = maete;
        MLRSig_OOF(test_idx)   = y_test_pred_SIG;

        % ---------------- RANDOM FOREST ----------------
        rng(10000 + rep * 100 + fold);

        RF_model = TreeBagger(numTrees, X_train, y_train, ...
            'Method', 'regression', ...
            'MinLeafSize', minLeafSize, ...
            'NumPredictorsToSample', size(X_train, 2), ...
            'OOBPrediction', 'off');

        y_train_pred_RF = predict(RF_model, X_train);
        y_test_pred_RF  = predict(RF_model, X_test);

        [r2tr, rmsetr, maetr] = calculate_metrics(y_train, y_train_pred_RF);
        [r2te, rmsete, maete] = calculate_metrics(y_test,  y_test_pred_RF);

        RF_R2_Train(idx)   = r2tr;
        RF_R2_Test(idx)    = r2te;
        RF_RMSE_Train(idx) = rmsetr;
        RF_RMSE_Test(idx)  = rmsete;
        RF_MAE_Train(idx)  = maetr;
        RF_MAE_Test(idx)   = maete;
        RF_OOF(test_idx)   = y_test_pred_RF;

        % ---------------- ENSEMBLE ----------------
        y_train_pred_ENS = MLR_weight * y_train_pred_MLR + RF_weight * y_train_pred_RF;
        y_test_pred_ENS  = MLR_weight * y_test_pred_MLR  + RF_weight * y_test_pred_RF;

        [r2tr, rmsetr, maetr] = calculate_metrics(y_train, y_train_pred_ENS);
        [r2te, rmsete, maete] = calculate_metrics(y_test,  y_test_pred_ENS);

        ENS_R2_Train(idx)   = r2tr;
        ENS_R2_Test(idx)    = r2te;
        ENS_RMSE_Train(idx) = rmsetr;
        ENS_RMSE_Test(idx)  = rmsete;
        ENS_MAE_Train(idx)  = maetr;
        ENS_MAE_Test(idx)   = maete;
        ENS_OOF(test_idx)   = y_test_pred_ENS;

    end
end

fprintf('\nCross-validation completed successfully.\n');

%% ========================================================================
% 17. SUMMARY STATISTICS
% ========================================================================

Summary_MLR_Full = make_summary('MLR-FULL', ...
    MLRFull_R2_Train, MLRFull_R2_Test, ...
    MLRFull_RMSE_Train, MLRFull_RMSE_Test, ...
    MLRFull_MAE_Train, MLRFull_MAE_Test);

Summary_MLR_Sig = make_summary('MLR-SIG', ...
    MLRSig_R2_Train, MLRSig_R2_Test, ...
    MLRSig_RMSE_Train, MLRSig_RMSE_Test, ...
    MLRSig_MAE_Train, MLRSig_MAE_Test);

Summary_RF = make_summary('RF-FULL', ...
    RF_R2_Train, RF_R2_Test, ...
    RF_RMSE_Train, RF_RMSE_Test, ...
    RF_MAE_Train, RF_MAE_Test);

Summary_Ensemble = make_summary('ENSEMBLE', ...
    ENS_R2_Train, ENS_R2_Test, ...
    ENS_RMSE_Train, ENS_RMSE_Test, ...
    ENS_MAE_Train, ENS_MAE_Test);

%% ========================================================================
% 18. DISPLAY RESULTS
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('                 CROSS-VALIDATION RESULTS\n');
fprintf('====================================================================\n');

print_summary(Summary_MLR_Full);
print_summary(Summary_MLR_Sig);
print_summary(Summary_RF);
print_summary(Summary_Ensemble);

%% ========================================================================
% 19. FINAL MODELS USING 100% OF DATA
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('                    FINAL MODELS\n');
fprintf('====================================================================\n');

% Final MLR-FULL
tbl_full_all = array2table(X, 'VariableNames', predictors);
tbl_full_all.(target) = y;
formula_full_all = sprintf('%s ~ %s', target, strjoin(predictors, ' + '));
Final_MLR_Full = fitlm(tbl_full_all, formula_full_all);

% Final MLR-SIG
if isempty(significant_predictors)
    Final_MLR_Sig = mdl_MLR_Sig_All;
else
    tbl_sig_all = array2table(X_sig, 'VariableNames', significant_predictors);
    tbl_sig_all.(target) = y;
    formula_sig_all = sprintf('%s ~ %s', target, strjoin(significant_predictors, ' + '));
    Final_MLR_Sig = fitlm(tbl_sig_all, formula_sig_all);
end

% Final Random Forest
rng(12345);
Final_RF = TreeBagger(numTrees, X, y, ...
    'Method', 'regression', ...
    'MinLeafSize', minLeafSize, ...
    'NumPredictorsToSample', size(X, 2), ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

fprintf('\nFinal models fitted using 100%% of valid data.\n');

%% ========================================================================
% 20. FINAL MODEL PERFORMANCE ON FULL DATA
% ========================================================================

pred_MLR_Full_all = predict(Final_MLR_Full, tbl_full_all);
[r2_MLR_Full_app, rmse_MLR_Full_app, mae_MLR_Full_app] = ...
    calculate_metrics(y, pred_MLR_Full_all);

if isempty(significant_predictors)
    pred_MLR_Sig_all = predict(Final_MLR_Sig, tbl_full_all);
else
    pred_MLR_Sig_all = predict(Final_MLR_Sig, tbl_sig_all);
end
[r2_MLR_Sig_app, rmse_MLR_Sig_app, mae_MLR_Sig_app] = ...
    calculate_metrics(y, pred_MLR_Sig_all);

pred_RF_all = predict(Final_RF, X);
[r2_RF_app, rmse_RF_app, mae_RF_app] = ...
    calculate_metrics(y, pred_RF_all);

pred_ENS_all = MLR_weight * pred_MLR_Full_all + RF_weight * pred_RF_all;
[r2_ENS_app, rmse_ENS_app, mae_ENS_app] = ...
    calculate_metrics(y, pred_ENS_all);

%% ========================================================================
% 21. PRINT APPARENT PERFORMANCE
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('           FULL-DATA APPARENT PERFORMANCE\n');
fprintf('====================================================================\n');

fprintf('\nNOTE: These are apparent/training statistics, not independent validation.\n');

fprintf('\nMLR-FULL:\n');
fprintf('R2   = %.4f\n', r2_MLR_Full_app);
fprintf('RMSE = %.4f\n', rmse_MLR_Full_app);
fprintf('MAE  = %.4f\n', mae_MLR_Full_app);

fprintf('\nMLR-SIG:\n');
fprintf('R2   = %.4f\n', r2_MLR_Sig_app);
fprintf('RMSE = %.4f\n', rmse_MLR_Sig_app);
fprintf('MAE  = %.4f\n', mae_MLR_Sig_app);

fprintf('\nRF-FULL:\n');
fprintf('R2   = %.4f\n', r2_RF_app);
fprintf('RMSE = %.4f\n', rmse_RF_app);
fprintf('MAE  = %.4f\n', mae_RF_app);

fprintf('\nENSEMBLE:\n');
fprintf('R2   = %.4f\n', r2_ENS_app);
fprintf('RMSE = %.4f\n', rmse_ENS_app);
fprintf('MAE  = %.4f\n', mae_ENS_app);

%% ========================================================================
% 22. RANDOM FOREST VARIABLE IMPORTANCE
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('              RANDOM FOREST VARIABLE IMPORTANCE\n');
fprintf('====================================================================\n');

RF_raw_importance = Final_RF.OOBPermutedPredictorDeltaError(:);
RF_positive_importance = max(RF_raw_importance, 0);

if sum(RF_positive_importance) > 0
    RF_importance_percent = 100 * RF_positive_importance ./ sum(RF_positive_importance);
else
    RF_importance_percent = zeros(size(RF_raw_importance));
end

RF_Importance = table();
RF_Importance.Predictor                  = predictors;
RF_Importance.OOB_Permutation_Importance = RF_raw_importance;
RF_Importance.Importance_Percent         = RF_importance_percent;
RF_Importance = sortrows(RF_Importance, 'Importance_Percent', 'descend');

disp(RF_Importance);

%% ========================================================================
% 23. MLR-FULL IMPORTANCE TABLE
% ========================================================================

MLR_Full_Importance = table();
MLR_Full_Importance.Predictor                   = predictors;
MLR_Full_Importance.Standardized_Beta           = Beta_full;
MLR_Full_Importance.Absolute_Standardized_Beta  = abs_beta_full;
MLR_Full_Importance.Importance_Percent          = Importance_full;
MLR_Full_Importance.pValue                      = p_predictors_full;
MLR_Full_Importance = sortrows(MLR_Full_Importance, 'Importance_Percent', 'descend');

%% ========================================================================
% 24. MLR-SIG IMPORTANCE TABLE
% ========================================================================

if ~isempty(significant_predictors)
    MLR_Sig_Importance = table();
    MLR_Sig_Importance.Predictor                  = significant_predictors;
    MLR_Sig_Importance.Standardized_Beta          = Beta_sig;
    MLR_Sig_Importance.Absolute_Standardized_Beta = abs(Beta_sig);
    MLR_Sig_Importance.Importance_Percent         = Importance_sig;
    MLR_Sig_Importance.pValue                     = pValue_sig(2:end);
    MLR_Sig_Importance = sortrows(MLR_Sig_Importance, 'Importance_Percent', 'descend');
else
    MLR_Sig_Importance = table();
end

%% ========================================================================
% 25. COMBINED IMPORTANCE TABLE
% ========================================================================

Combined_Importance = table();
Combined_Importance.Predictor                      = predictors;
Combined_Importance.MLR_Standardized_Beta          = Beta_full;
Combined_Importance.MLR_Importance_Percent         = Importance_full;
Combined_Importance.MLR_pValue                     = p_predictors_full;
Combined_Importance.RF_OOB_Permutation_Importance  = RF_raw_importance;
Combined_Importance.RF_Importance_Percent          = RF_importance_percent;

%% ========================================================================
% 26. MODEL COMPARISON TABLE
% ========================================================================

Model_Comparison = table();
Model_Comparison.Model = {'MLR-FULL'; 'MLR-SIG'; 'RF-FULL'; 'ENSEMBLE'};

Model_Comparison.Validation_R2_Mean   = [Summary_MLR_Full.Validation_R2_Mean; ...
                                         Summary_MLR_Sig.Validation_R2_Mean; ...
                                         Summary_RF.Validation_R2_Mean; ...
                                         Summary_Ensemble.Validation_R2_Mean];
Model_Comparison.Validation_R2_SD     = [Summary_MLR_Full.Validation_R2_SD; ...
                                         Summary_MLR_Sig.Validation_R2_SD; ...
                                         Summary_RF.Validation_R2_SD; ...
                                         Summary_Ensemble.Validation_R2_SD];
Model_Comparison.Validation_RMSE_Mean = [Summary_MLR_Full.Validation_RMSE_Mean; ...
                                         Summary_MLR_Sig.Validation_RMSE_Mean; ...
                                         Summary_RF.Validation_RMSE_Mean; ...
                                         Summary_Ensemble.Validation_RMSE_Mean];
Model_Comparison.Validation_RMSE_SD   = [Summary_MLR_Full.Validation_RMSE_SD; ...
                                         Summary_MLR_Sig.Validation_RMSE_SD; ...
                                         Summary_RF.Validation_RMSE_SD; ...
                                         Summary_Ensemble.Validation_RMSE_SD];
Model_Comparison.Validation_MAE_Mean  = [Summary_MLR_Full.Validation_MAE_Mean; ...
                                         Summary_MLR_Sig.Validation_MAE_Mean; ...
                                         Summary_RF.Validation_MAE_Mean; ...
                                         Summary_Ensemble.Validation_MAE_Mean];
Model_Comparison.Validation_MAE_SD    = [Summary_MLR_Full.Validation_MAE_SD; ...
                                         Summary_MLR_Sig.Validation_MAE_SD; ...
                                         Summary_RF.Validation_MAE_SD; ...
                                         Summary_Ensemble.Validation_MAE_SD];

%% ========================================================================
% 27. ALL FOLD RESULTS
% ========================================================================

CV_All_Folds = table();
CV_All_Folds.Repetition = RepNumber;
CV_All_Folds.Fold       = FoldNumber;

CV_All_Folds.MLR_Full_R2_Train            = MLRFull_R2_Train;
CV_All_Folds.MLR_Full_R2_Validation       = MLRFull_R2_Test;
CV_All_Folds.MLR_Full_RMSE_Train          = MLRFull_RMSE_Train;
CV_All_Folds.MLR_Full_RMSE_Validation     = MLRFull_RMSE_Test;
CV_All_Folds.MLR_Full_MAE_Train           = MLRFull_MAE_Train;
CV_All_Folds.MLR_Full_MAE_Validation      = MLRFull_MAE_Test;

CV_All_Folds.MLR_Sig_R2_Train             = MLRSig_R2_Train;
CV_All_Folds.MLR_Sig_R2_Validation        = MLRSig_R2_Test;
CV_All_Folds.MLR_Sig_RMSE_Train           = MLRSig_RMSE_Train;
CV_All_Folds.MLR_Sig_RMSE_Validation      = MLRSig_RMSE_Test;
CV_All_Folds.MLR_Sig_MAE_Train            = MLRSig_MAE_Train;
CV_All_Folds.MLR_Sig_MAE_Validation       = MLRSig_MAE_Test;

CV_All_Folds.RF_R2_Train                  = RF_R2_Train;
CV_All_Folds.RF_R2_Validation             = RF_R2_Test;
CV_All_Folds.RF_RMSE_Train                = RF_RMSE_Train;
CV_All_Folds.RF_RMSE_Validation           = RF_RMSE_Test;
CV_All_Folds.RF_MAE_Train                 = RF_MAE_Train;
CV_All_Folds.RF_MAE_Validation            = RF_MAE_Test;

CV_All_Folds.Ensemble_R2_Train            = ENS_R2_Train;
CV_All_Folds.Ensemble_R2_Validation       = ENS_R2_Test;
CV_All_Folds.Ensemble_RMSE_Train          = ENS_RMSE_Train;
CV_All_Folds.Ensemble_RMSE_Validation     = ENS_RMSE_Test;
CV_All_Folds.Ensemble_MAE_Train           = ENS_MAE_Train;
CV_All_Folds.Ensemble_MAE_Validation      = ENS_MAE_Test;

%% ========================================================================
% 28. OUT-OF-FOLD PREDICTIONS
% ========================================================================

OOF_Predictions = table();
OOF_Predictions.Observed            = y;
OOF_Predictions.MLR_Full_Predicted  = MLRFull_OOF;
OOF_Predictions.MLR_Sig_Predicted   = MLRSig_OOF;
OOF_Predictions.RF_Full_Predicted   = RF_OOF;
OOF_Predictions.Ensemble_Predicted  = ENS_OOF;

%% ========================================================================
% 29. OOF OVERALL METRICS
% ========================================================================

[OOF_R2_MLR_Full,  OOF_RMSE_MLR_Full,  OOF_MAE_MLR_Full]  = calculate_metrics(y, MLRFull_OOF);
[OOF_R2_MLR_Sig,   OOF_RMSE_MLR_Sig,   OOF_MAE_MLR_Sig]   = calculate_metrics(y, MLRSig_OOF);
[OOF_R2_RF,        OOF_RMSE_RF,        OOF_MAE_RF]        = calculate_metrics(y, RF_OOF);
[OOF_R2_ENS,       OOF_RMSE_ENS,       OOF_MAE_ENS]       = calculate_metrics(y, ENS_OOF);

OOF_Summary = table();
OOF_Summary.Model = {'MLR-FULL'; 'MLR-SIG'; 'RF-FULL'; 'ENSEMBLE'};
OOF_Summary.R2    = [OOF_R2_MLR_Full;  OOF_R2_MLR_Sig;  OOF_R2_RF;  OOF_R2_ENS];
OOF_Summary.RMSE  = [OOF_RMSE_MLR_Full; OOF_RMSE_MLR_Sig; OOF_RMSE_RF; OOF_RMSE_ENS];
OOF_Summary.MAE   = [OOF_MAE_MLR_Full;  OOF_MAE_MLR_Sig;  OOF_MAE_RF;  OOF_MAE_ENS];

%% ========================================================================
% 30. SIGNIFICANCE TABLE
% ========================================================================

Significance_Table = table();
Significance_Table.Predictor          = predictors;
Significance_Table.Coefficient_B      = Estimate_full(2:end);
Significance_Table.Standardized_Beta  = Beta_full;
Significance_Table.pValue             = p_predictors_full;
Significance_Table.Significant_at_005 = significant_mask;
Significance_Table.Status             = cell(length(predictors), 1);

for i = 1:length(predictors)
    if significant_mask(i)
        Significance_Table.Status{i} = 'Significant';
    else
        Significance_Table.Status{i} = 'Not Significant';
    end
end

%% ========================================================================
% 31. SAVE ALL RESULTS TO ONE EXCEL FILE
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('                    SAVING EXCEL OUTPUT\n');
fprintf('====================================================================\n');

excel_file = fullfile(output_dir, sprintf('FINAL_SOIL_THERMAL_%s.xlsx', target));

writetable(Model_Comparison,          excel_file, 'Sheet', 'Model_Comparison');
writetable(Summary_MLR_Full,          excel_file, 'Sheet', 'MLR_Full_CV');
writetable(Summary_MLR_Sig,           excel_file, 'Sheet', 'MLR_Sig_CV');
writetable(Summary_RF,                excel_file, 'Sheet', 'RF_Full_CV');
writetable(Summary_Ensemble,          excel_file, 'Sheet', 'Ensemble_CV');
writetable(CV_All_Folds,              excel_file, 'Sheet', 'CV_All_Folds');
writetable(MLR_Full_Coefficients,     excel_file, 'Sheet', 'MLR_Full_Coefficients');
writetable(MLR_Sig_Coefficients,      excel_file, 'Sheet', 'MLR_Sig_Coefficients');
writetable(MLR_Full_Importance,       excel_file, 'Sheet', 'MLR_Full_Importance');
writetable(MLR_Sig_Importance,        excel_file, 'Sheet', 'MLR_Sig_Importance');
writetable(RF_Importance,             excel_file, 'Sheet', 'RF_Importance');
writetable(Combined_Importance,       excel_file, 'Sheet', 'Combined_Importance');
writetable(Significance_Table,        excel_file, 'Sheet', 'Significance');
writetable(OOF_Predictions,           excel_file, 'Sheet', 'OOF_Predictions');
writetable(OOF_Summary,               excel_file, 'Sheet', 'OOF_Summary');

fprintf('\nExcel file saved:\n%s\n', excel_file);

%% ========================================================================
% 32. SAVE MATLAB MODELS
% ========================================================================

mat_file = fullfile(output_dir, sprintf('FINAL_MODELS_%s.mat', target));

save(mat_file, ...
    'Final_MLR_Full', 'Final_MLR_Sig', 'Final_RF', ...
    'significant_predictors', 'non_significant_predictors', ...
    'predictors', 'target', 'MLR_weight', 'RF_weight', ...
    'numTrees', 'minLeafSize', 'fold_assignment');

fprintf('\nMATLAB models saved:\n%s\n', mat_file);

%% ========================================================================
% 33. SAVE OOF PREDICTIONS AND MODEL COMPARISON AS SEPARATE FILES
% ========================================================================

oof_file = fullfile(output_dir, sprintf('OOF_Predictions_%s.xlsx', target));
writetable(OOF_Predictions, oof_file);
fprintf('\nOOF predictions saved:\n%s\n', oof_file);

summary_file = fullfile(output_dir, sprintf('Model_Comparison_%s.xlsx', target));
writetable(Model_Comparison, summary_file);
fprintf('\nModel comparison saved:\n%s\n', summary_file);

%% ========================================================================
% 34. FINAL SUMMARY
% ========================================================================

fprintf('\n');
fprintf('====================================================================\n');
fprintf('                        FINAL SUMMARY\n');
fprintf('====================================================================\n');

fprintf('\nTarget: %s\n', target);

fprintf('\nSignificant predictors (p < %.2f):\n', alpha);
if isempty(significant_predictors)
    fprintf('   None\n');
else
    for i = 1:length(significant_predictors)
        fprintf('   %s\n', significant_predictors{i});
    end
end

fprintf('\nValidation performance: Mean ± SD\n');

fprintf('\nMLR-FULL\n');
fprintf('R2   : %.4f ± %.4f\n', Summary_MLR_Full.Validation_R2_Mean,   Summary_MLR_Full.Validation_R2_SD);
fprintf('RMSE : %.4f ± %.4f\n', Summary_MLR_Full.Validation_RMSE_Mean, Summary_MLR_Full.Validation_RMSE_SD);
fprintf('MAE  : %.4f ± %.4f\n', Summary_MLR_Full.Validation_MAE_Mean,  Summary_MLR_Full.Validation_MAE_SD);

fprintf('\nMLR-SIG\n');
fprintf('R2   : %.4f ± %.4f\n', Summary_MLR_Sig.Validation_R2_Mean,   Summary_MLR_Sig.Validation_R2_SD);
fprintf('RMSE : %.4f ± %.4f\n', Summary_MLR_Sig.Validation_RMSE_Mean, Summary_MLR_Sig.Validation_RMSE_SD);
fprintf('MAE  : %.4f ± %.4f\n', Summary_MLR_Sig.Validation_MAE_Mean,  Summary_MLR_Sig.Validation_MAE_SD);

fprintf('\nRF-FULL\n');
fprintf('R2   : %.4f ± %.4f\n', Summary_RF.Validation_R2_Mean,   Summary_RF.Validation_R2_SD);
fprintf('RMSE : %.4f ± %.4f\n', Summary_RF.Validation_RMSE_Mean, Summary_RF.Validation_RMSE_SD);
fprintf('MAE  : %.4f ± %.4f\n', Summary_RF.Validation_MAE_Mean,  Summary_RF.Validation_MAE_SD);

fprintf('\nENSEMBLE\n');
fprintf('R2   : %.4f ± %.4f\n', Summary_Ensemble.Validation_R2_Mean,   Summary_Ensemble.Validation_R2_SD);
fprintf('RMSE : %.4f ± %.4f\n', Summary_Ensemble.Validation_RMSE_Mean, Summary_Ensemble.Validation_RMSE_SD);
fprintf('MAE  : %.4f ± %.4f\n', Summary_Ensemble.Validation_MAE_Mean,  Summary_Ensemble.Validation_MAE_SD);

fprintf('\n');
fprintf('====================================================================\n');
fprintf('                   MODELING COMPLETED SUCCESSFULLY\n');
fprintf('====================================================================\n');
fprintf('\nOutput directory:\n%s\n', output_dir);
fprintf('\nMain Excel file:\n%s\n', excel_file);
fprintf('\n');

%% ========================================================================
%                         LOCAL FUNCTIONS
% ========================================================================

function [R2, RMSE, MAE] = calculate_metrics(y_true, y_pred)

    y_true = y_true(:);
    y_pred = y_pred(:);

    valid = ~isnan(y_true) & ~isnan(y_pred);
    y_true = y_true(valid);
    y_pred = y_pred(valid);

    denominator = sum((y_true - mean(y_true)).^2);
    if denominator == 0
        R2 = NaN;
    else
        R2 = 1 - sum((y_true - y_pred).^2) / denominator;
    end

    RMSE = sqrt(mean((y_true - y_pred).^2));
    MAE  = mean(abs(y_true - y_pred));

end


function Summary = make_summary(model_name, ...
    R2_train, R2_test, RMSE_train, RMSE_test, MAE_train, MAE_test)

    Summary = table();
    Summary.Model = {model_name};

    Summary.Train_R2_Mean       = mean(R2_train, 'omitnan');
    Summary.Train_R2_SD         = std(R2_train, 'omitnan');
    Summary.Validation_R2_Mean  = mean(R2_test, 'omitnan');
    Summary.Validation_R2_SD    = std(R2_test, 'omitnan');

    Summary.Train_RMSE_Mean     = mean(RMSE_train, 'omitnan');
    Summary.Train_RMSE_SD       = std(RMSE_train, 'omitnan');
    Summary.Validation_RMSE_Mean = mean(RMSE_test, 'omitnan');
    Summary.Validation_RMSE_SD   = std(RMSE_test, 'omitnan');

    Summary.Train_MAE_Mean      = mean(MAE_train, 'omitnan');
    Summary.Train_MAE_SD        = std(MAE_train, 'omitnan');
    Summary.Validation_MAE_Mean = mean(MAE_test, 'omitnan');
    Summary.Validation_MAE_SD   = std(MAE_test, 'omitnan');

end


function print_summary(Summary)

    fprintf('\n');
    fprintf('--------------------------------------------------------------------\n');
    fprintf('%s\n', Summary.Model{1});
    fprintf('--------------------------------------------------------------------\n');
    fprintf('Metric       | Train Mean | Train SD | Validation Mean | Validation SD\n');
    fprintf('--------------------------------------------------------------------\n');

    fprintf('R2           | %.4f     | %.4f   | %.4f          | %.4f\n', ...
        Summary.Train_R2_Mean, Summary.Train_R2_SD, ...
        Summary.Validation_R2_Mean, Summary.Validation_R2_SD);

    fprintf('RMSE         | %.4f     | %.4f   | %.4f          | %.4f\n', ...
        Summary.Train_RMSE_Mean, Summary.Train_RMSE_SD, ...
        Summary.Validation_RMSE_Mean, Summary.Validation_RMSE_SD);

    fprintf('MAE          | %.4f     | %.4f   | %.4f          | %.4f\n', ...
        Summary.Train_MAE_Mean, Summary.Train_MAE_SD, ...
        Summary.Validation_MAE_Mean, Summary.Validation_MAE_SD);

end
