import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import scipy
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
import time
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.optimize import minimize
from collections import Counter
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

def set_global_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

def describe_x(df):
    X = df['X']
    return [
        X.std(),
    ]

def describe_y(df):
    Y = df['Y']
    return [
        Y.std(),
    ]

def describe_z(df):
    Z = df['Z']
    return [
        Z.std(),  
    ]

def describe_enmo(df):
    enmo = df['enmo']
    return [
        enmo.mean(),  
    ]

def describe_anglez(df):
    anglez = df['anglez']
    return [
        anglez.std(),
    ]
    
# Light level thresholds (in lux)
light_bins = [
    (0, 5, 'Twilight'),
    (5, 10, 'Minimal Street Lighting'),
    (10, 50, 'Sunset'),
    (50, 80, 'Family Living Room'),
    (80, 100, 'Hallway'),
    (100, 320, 'Very Dark Overcast Day'),
    (320, 500, 'Office Lighting'),
    (500, 1000, 'Sunrise/Sunset'),
    (1000, 10000, 'Overcast Day'),
    (10000, 25000, 'Full Daylight'),
    (25000, 130000, 'Direct Sunlight')
]


def categorize_light(light_value):
    for low, high, label in light_bins:
        if low <= light_value < high:
            return label
    return 'Unknown'

def describe_light(df):
    df['light_category'] = df['light'].apply(categorize_light)
    light_categories = df['light_category'].value_counts(normalize=True).to_dict()
    
    features = [light_categories.get(label, 0) for _, _, label in light_bins]
    return features

def longest_inactivity_streaks(df, window_size=100, threshold=10, top_n=5):
    rolling_cumsum = df['enmo'].rolling(window=window_size).sum()
    inactive = rolling_cumsum <= threshold
    
    # Calculate streaks
    streak_lengths = []
    current_streak = 0
    for is_inactive in inactive:
        if is_inactive:
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_streak = 0
    
    # If the last streak is still active, add it
    if current_streak > 0:
        streak_lengths.append(current_streak)
    
    # Sort streaks in descending order and pick top N
    streak_lengths = sorted(streak_lengths, reverse=True)[:top_n]
    
    # Pad with zeros if there are fewer than N streaks
    streak_lengths += [0] * (top_n - len(streak_lengths))
    return streak_lengths


def longest_activity_streaks(df, window_size=100, threshold=1, top_n=5):
    # Calculate cumsum of enmo in the defined window
    rolling_cumsum = df['enmo'].rolling(window=window_size).sum()
    
    # Identify active windows (cumsum > threshold)
    active = rolling_cumsum > threshold
    
    # Calculate streaks
    streak_lengths = []
    current_streak = 0
    for is_active in active:
        if is_active:
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_streak = 0
    
    # If the last streak is still active, add it
    if current_streak > 0:
        streak_lengths.append(current_streak)
    
    # Sort streaks in descending order and pick top N
    streak_lengths = sorted(streak_lengths, reverse=True)[:top_n]
    
    # Pad with zeros if there are fewer than N streaks
    streak_lengths += [0] * (top_n - len(streak_lengths))
    return streak_lengths


def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop(['step'], axis=1, inplace=True)
   
    features = []
    features.extend(describe_x(df))
    features.extend(describe_y(df))
    features.extend(describe_z(df))
    features.extend(describe_enmo(df))
    features.extend(describe_anglez(df))
    features.extend(describe_light(df))  
    
    enmo_active_ratio = (df['enmo'] > 0).mean()
    features.append(enmo_active_ratio)
    features.extend(longest_inactivity_streaks(df, threshold=1))
    features.extend(longest_activity_streaks(df, threshold=5))
   
    return np.array(features), filename.split('=')[1]



def load_time_series(path) -> pd.DataFrame:
    # for kaggle folder
    if os.path.isdir(path):
        ids = os.listdir(path)
        if not ids:
            print(f"La carpeta {path} está vacía.")
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda fname: process_file(fname, path), ids), total=len(ids)))
        if results:
            stats, indexes = zip(*results)
            df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
            df['id'] = indexes
            return df
        else:
            raise ValueError(f"No se encontraron archivos para procesar en {path}")
    # for unique parquet file
    elif os.path.isfile(path) and path.endswith('.parquet'):
        print(f"Leyendo archivo único: {path}")
        df = pd.read_parquet(path)
        features = []
        features.extend(describe_x(df))
        features.extend(describe_y(df))
        features.extend(describe_z(df))
        features.extend(describe_enmo(df))
        features.extend(describe_anglez(df))
        features.extend(describe_light(df))
        enmo_active_ratio = (df['enmo'] > 0).mean()
        features.append(enmo_active_ratio)
        features.extend(longest_inactivity_streaks(df, threshold=1))
        features.extend(longest_activity_streaks(df, threshold=5))
        # uncomment if you want to return the id
        return pd.DataFrame([features], columns=[f"stat_{i}" for i in range(len(features))])
    else:
        raise ValueError(f"Ruta no válida: {path}")

train_ts = load_time_series("series_train.parquet")
test_ts = load_time_series("series_test.parquet")

train_ts

def feature_engineering(df):

    for col, (col_min, col_max) in min_max_dict.items():
        df[col] = df[col].clip(lower=col_min, upper=col_max)

    bins = [0, 6, 12, 18, 100]
    labels = ['1 to 6', '7 to 12', '13 to 18', '19 to 100']
    df['Age_Binned'] = pd.cut(df['Basic_Demos-Age'], bins=bins, labels=labels, right=True)
    df['Age_Sex'] = df['Age_Binned'].astype(str) + '_' + df['Basic_Demos-Sex'].astype(str)
    
    df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
    df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
    df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
    
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    
    df['PreInt_FGC_CU_PU'] = df['PreInt_EduHx-computerinternet_hoursday'] * df['FGC-FGC_CU'] * df['FGC-FGC_PU']
    df['FGC_GSND_GSD_Age'] = df['FGC-FGC_GSND'] * df['FGC-FGC_GSD'] * df['Basic_Demos-Age']
    df['SDS_Activity'] = df['BIA-BIA_Activity_Level_num'] * df['SDS-SDS_Total_T']
    
    df['CGasync_Score_Normalized'] = df['CGAS-CGAS_Score'] - df.groupby('Basic_Demos-Enroll_Season')['CGAS-CGAS_Score'].transform('mean')
    df['Internet_Physical_Difference'] = df['PreInt_EduHx-computerinternet_hoursday'] - df['PAQ_A-PAQ_A_Total']
   
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').astype('category')
    return df

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')


numeric_cols = train[test.columns].select_dtypes(include='number').columns
min_max_dict = {col: (train[col].min(), train[col].max()) for col in numeric_cols}

train = feature_engineering(train)
test = feature_engineering(test)

train = train.drop('id', axis=1)
test  = test .drop('id', axis=1)   

train = train.dropna(subset='sii')

target = train['PCIAT-PCIAT_Total']
sii_target = train['sii']
train = train[test.columns]

def map_pciat_to_sii(pciat_values):
    return np.select(
        [pciat_values <= 30, 
         (pciat_values > 30) & (pciat_values <= 49),
         (pciat_values > 49) & (pciat_values <= 79),
         pciat_values > 79],
        [0, 1, 2, 3],
        default=3  # For PCIAT values greater than 79
    )
    
def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
def select_subset(df, target, subset_size=0.8):
    df_subset = df.sample(frac=subset_size, random_state=42)
    target_subset = target.loc[df_subset.index]
    return df_subset, target_subset


def gaussian_noise_injection(df, target, noise_level, subset_size=0.2):

    # Select a subset of data for augmentation
    df_subset, target_subset = select_subset(df, target, subset_size)

    # Split numeric and non-numeric columns
    numeric_cols = df_subset.select_dtypes(include=['float64', 'int64'])
    non_numeric_cols = df_subset.select_dtypes(exclude=['float64', 'int64'])

    # Impute missing values in numeric columns
    imputer = SimpleImputer(strategy='mean')
    numeric_imputed = pd.DataFrame(imputer.fit_transform(numeric_cols), 
                                   columns=numeric_cols.columns, 
                                   index=numeric_cols.index)

    # Add noise to numeric columns
    augmented_numeric = numeric_imputed
    for col in augmented_numeric.columns:
        std_dev = augmented_numeric[col].std()
        if std_dev > 0:  # Add noise only if variability exists
            noise = np.random.normal(0, noise_level * std_dev, size=len(augmented_numeric))
            augmented_numeric[col] += noise

    # Concatenate back with non-numeric columns (align rows)
    augmented_df = pd.concat([augmented_numeric, non_numeric_cols], axis=1)

    # Ensure the column order matches the original subset
    augmented_df = augmented_df[df_subset.columns]
    return augmented_df, target_subset

def augment_data_with_nans(X, target, threshold=0.1, subset_size=0.2):
   
    df_subset, target_subset = select_subset(X, target, subset_size)
    X_augmented = df_subset.reset_index(drop=True).copy()
    
    # Identify columns that already contain NaN values
    columns_with_nan = [col for col in X.columns if X[col].isna().sum() > 0]
    
    # Mask for non-NaN values in columns that contain NaNs
    non_nan_mask = X_augmented[columns_with_nan].notna()
    
    # Randomly select which column to set to NaN (for each row) where there's a valid value
    for col in columns_with_nan:
        # Create a random mask for columns with valid values (non-NaN)
        random_mask = np.random.rand(len(X_augmented)) < threshold  # Adjust probability as needed
        
        # Apply the mask to select rows and set that column's value to NaN
        X_augmented.loc[random_mask, col] = np.nan
    
    return X_augmented, target_subset

def plot_confusion_matrix(y_true, y_pred, labels=None):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    
    if labels is None:
        labels = sorted(set(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def cross_validate_model(params, X, y, sii_target, label='', save_models=True, pruning_callback=None, n_repeats=5, return_qwk=False):
    features = X.columns
    start_time = time.time()
    oof = []
    y_oof = []
    qwk_list = []
    model_list = []
   
    n = 0
    for repeat in tqdm(range(n_repeats)):
        random_seed = np.random.randint(0, 10000)  # Generate a random seed for each repeat
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
        
        for fold, (idx_tr, idx_va) in enumerate(folds.split(X, sii_target)):
            params['random_seeds'] = n
            set_global_seed(n)
            X_tr = X.iloc[idx_tr]
            X_va = X.iloc[idx_va]
            y_tr = y.iloc[idx_tr]
            y_va = y.iloc[idx_va]
            
            
            nan_prone_columns = [
                col for col in X_tr.columns 
                if X_tr[col].isna().any()  # Has NaNs
            ]
    
            # Step 1: Perform augmentation on X_tr
            nan_augmented, nan_aug_target = augment_data_with_nans(X_tr, target, threshold=1, subset_size=0.2)
            noise_augmented, noise_aug_target = gaussian_noise_injection(X_tr, y_tr, noise_level=0.02, subset_size=0.5)
    
            X_tr_augmented = pd.concat(
                [nan_augmented, noise_augmented, X_tr[y_tr>49], X_tr[y_tr>49], X_tr[y_tr>49], X_tr[y_tr>79]],
                ignore_index=True).reset_index(drop=True)
            
            y_tr_augmented = pd.concat(
                [nan_aug_target, noise_aug_target, y_tr[y_tr>49], y_tr[y_tr>49], y_tr[y_tr>49], y_tr[y_tr>79]],
                ignore_index=True).reset_index(drop=True)


            X_tr_combined = pd.concat([X_tr, X_tr_augmented], ignore_index=True).reset_index(drop=True)
            y_tr_combined = pd.concat([y_tr, y_tr_augmented], ignore_index=True).reset_index(drop=True)

            shuffled_indices = np.random.permutation(X_tr_combined.index)
            X_tr_combined = X_tr_combined.iloc[shuffled_indices].reset_index(drop=True)
            y_tr_combined = y_tr_combined.iloc[shuffled_indices].reset_index(drop=True)

            
            dtrain = lgb.Dataset(X_tr_combined, label=y_tr_combined)
            dvalid = lgb.Dataset(X_va, label=y_va)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dtrain, dvalid],
                num_boost_round=params['n_estimators'],
            )

            y_pred = model.predict(X_va)

            if save_models:
                model_list.append(model)
            oof.append(y_pred)
            y_oof.append(y_va)
            
            n +=1
    elapsed_time = time.time() - start_time

    y_oof_actuals = np.concatenate(y_oof)
    oof_preds = np.concatenate(oof)
    
    # Post-processing: Map predictions
    y_oof_sii = map_pciat_to_sii(y_oof_actuals)
    oof_sii = map_pciat_to_sii(oof_preds)

  
    qwk = cohen_kappa_score(y_oof_sii, oof_sii, weights='quadratic')
    mse = ((y_oof_actuals - oof_preds)**2).mean()  
    print(f"Overall QWK: {qwk:.3f}, MSE: {mse:.3f}, Time: {int((time.time() - start_time) / 60)} min")

    # Optimize thresholds
    threshold_optimizer = minimize(evaluate_predictions, 
                                   x0=[34, 49, 62], 
                                   args=(y_oof_sii, oof_preds), 
                                   method='Nelder-Mead')
    
    optimized_preds = threshold_Rounder(oof_preds, threshold_optimizer.x)
    optimized_qwk = cohen_kappa_score(y_oof_sii, optimized_preds, weights='quadratic')
    accuracy = (y_oof_sii==optimized_preds).astype(np.float32).mean()
    print(f"Optimized QWK: {optimized_qwk:.3f}, Accuracy: {accuracy:.3f}, Thresholds: {threshold_optimizer.x}")
    
    plot_confusion_matrix(y_oof_sii, oof_sii)
    
    if save_models:
        saved_models[label] = {'features': features, 'model_list': model_list}

    return optimized_qwk, threshold_optimizer.x

saved_models = {}
results = []
for i in range(1):
    params = {'verbosity': -1,  'device': 'cpu', 'metric': 'mse', 'n_estimators':150, 'max_depth':5, 'max_bin': 15, 'boosting_type': 'gbdt', 'lambda_l1': 0.0012071403780584485, 'lambda_l2': 19.943477818207878, 'min_child_weight': 0.01586977190723854, 'learning_rate': 0.030512450456770007, 'num_leaves': 295, 'colsample_bytree': 0.8569995659929517, 'bagging_fraction': 0.587037100215173, 'feature_fraction': 0.8955475330753205, 'bagging_freq': 1}
    qwk, qwk_thresholded = cross_validate_model(params, train, target, sii_target, label='trial', save_models=True, n_repeats=100)
    print(qwk)
    results.append(qwk)
print(f"'mean {np.mean(results)}")
print(f"diff {max(results) - min(results)}")

pred = [model.predict(test)  for model in saved_models['trial']['model_list']]

n = 16
i = 500
plt.hist(np.array(pred)[:, n][:i], bins=30, alpha=0.7)

# Get the mode
mode_val = stats.mode(np.array(pred)[:, n][:i].round())[0]  # mode.value[0]

# Overlay the mode on the histogram
plt.axvline(mode_val, color='k', linestyle='dashed', linewidth=2, label=f'Mode: {mode_val}')
plt.axvline(np.array(pred)[:, n][:i].mean(), color='r', linestyle='dashed', linewidth=2, label=f'mean: {np.array(pred)[:, n][:i].mean()}')
# Add a label
plt.legend()

plt.show()

predictions = stats.mode(threshold_Rounder(np.array([model.predict(test) for model in saved_models['trial']['model_list']]), qwk_thresholded).astype(np.int32))[0]

submission_df = pd.read_csv('./sample_submission.csv')

submission_df['sii'] = predictions
submission_df.to_csv('submission.csv', index=False)
pd.read_csv('./submission.csv')