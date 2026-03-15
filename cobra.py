import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
from io import BytesIO
import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import pickle
import warnings
import os
from xgboost import XGBClassifier
import webbrowser
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Loading Data
df     = pd.read_csv("noaa_virtual_stations.csv")
coords = pd.read_csv("reef_coordinates.csv")

df['date'] = pd.to_datetime(df[['YYYY','MM','DD']].rename(
    columns={'YYYY':'year','MM':'month','DD':'day'}
))
df = df.sort_values(['reef','date']).reset_index(drop=True)
df.replace(-999.0,   np.nan, inplace=True)
df.replace(-32768.0, np.nan, inplace=True)
#-----------------------------------------------------------------

# Feature Engineering
def engineer_features(group):
    g          = group.copy()
    ssta       = g['SSTA_90th_HS']
    dhw        = g['DHW_from_90th_HS_gt_1']
    baa        = g['BAA_7day_max']
    hs         = g['90th_HS_gt_0']
    temp_range = g['SST_MAX'] - g['SST_MIN']

    g['ssta_current'] = ssta
    g['dhw_current']  = dhw
    g['temp_range']   = temp_range

    # rolling averages + peaks (up to 90 days to catch slow buildup)
    for signal, name in [(ssta, 'ssta'), (dhw, 'dhw'), (temp_range, 'temp_range')]:
        for w in [7, 14, 30, 60, 90]:
            g[f'{name}_{w}d_mean'] = signal.rolling(w, min_periods=1).mean()
            g[f'{name}_{w}d_max']  = signal.rolling(w, min_periods=1).max()
            g[f'{name}_{w}d_std']  = signal.rolling(w, min_periods=1).std()  # new: volatility

    # rate of change
    g['dhw_trend_7d']   = dhw.diff(7)
    g['dhw_trend_14d']  = dhw.diff(14)
    g['dhw_trend_30d']  = dhw.diff(30) 
    g['ssta_trend_7d']  = ssta.diff(7)  
    g['ssta_trend_14d'] = ssta.diff(14)

    # acceleration
    g['dhw_accel']  = g['dhw_trend_7d'].diff(7)
    g['ssta_accel'] = g['ssta_trend_7d'].diff(7)  # new: is anomaly speeding up?

    # dangerous days — extended to 60 days
    g['days_above_hotspot_30d'] = (ssta > 1).rolling(30, min_periods=1).sum()
    g['days_above_hotspot_60d'] = (ssta > 1).rolling(60, min_periods=1).sum()
    g['days_above_bleach_30d']  = (dhw  > 4).rolling(30, min_periods=1).sum()
    g['days_above_bleach_60d']  = (dhw  > 4).rolling(60, min_periods=1).sum() 
    g['days_hs_active_30d']     = (hs   > 0).rolling(30, min_periods=1).sum()
    g['days_ssta_gt2_30d']      = (ssta > 2).rolling(30, min_periods=1).sum() 

    # seasonality
    g['month_sin'] = np.sin(2 * np.pi * g['MM'] / 12)
    g['month_cos'] = np.cos(2 * np.pi * g['MM'] / 12)

    # NOAA alert history — extended lookback
    g['baa_lag_7d']  = baa.shift(7)
    g['baa_lag_14d'] = baa.shift(14)   
    g['baa_lag_30d'] = baa.shift(30)   
    g['baa_max_30d'] = baa.rolling(30, min_periods=1).max()
    g['baa_max_60d'] = baa.rolling(60, min_periods=1).max() 

    # target
    g['bleaching_in_6wks'] = baa.shift(-42).ge(3).astype(int)

    return g

df = df.groupby('reef', group_keys=False).apply(engineer_features)

FEATURES = [
    'ssta_current', 'dhw_current', 'temp_range',

    'ssta_7d_mean',  'ssta_14d_mean',  'ssta_30d_mean',  'ssta_60d_mean',  'ssta_90d_mean',
    'dhw_7d_mean',   'dhw_14d_mean',   'dhw_30d_mean',   'dhw_60d_mean',   'dhw_90d_mean',
    'temp_range_7d_mean', 'temp_range_14d_mean', 'temp_range_30d_mean',

    'ssta_7d_max',   'ssta_14d_max',   'ssta_30d_max',   'ssta_60d_max',   'ssta_90d_max',
    'dhw_7d_max',    'dhw_14d_max',    'dhw_30d_max',    'dhw_60d_max',    'dhw_90d_max',
    'temp_range_7d_max', 'temp_range_14d_max', 'temp_range_30d_max',

    'ssta_7d_std',   'ssta_30d_std',   'dhw_7d_std',     'dhw_30d_std',

    'dhw_trend_7d',  'dhw_trend_14d',  'dhw_trend_30d',
    'ssta_trend_7d', 'ssta_trend_14d',
    'dhw_accel',     'ssta_accel',

    'days_above_hotspot_30d', 'days_above_hotspot_60d',
    'days_above_bleach_30d',  'days_above_bleach_60d',
    'days_hs_active_30d',     'days_ssta_gt2_30d',

    'month_sin', 'month_cos',

    'baa_lag_7d', 'baa_lag_14d', 'baa_lag_30d',
    'baa_max_30d', 'baa_max_60d',
]

df[FEATURES] = df[FEATURES].fillna(0)

#-------------------------------------------------------------------------------------------------
#normalize each of the feature vectors and assign a weight to each one
#the normalizing process would consist of taking the mean and the standard deviation of each category and seeing where that specific data pointn fits in
#this would become the y and the x will be the dates

# split/train data
cutoff     = pd.Timestamp('2020-01-01')
train_mask = df['date'] < cutoff
test_mask  = df['date'] >= cutoff

X_train = df[train_mask][FEATURES]
y_train = df[train_mask]['bleaching_in_6wks']
X_test  = df[test_mask][FEATURES]
y_test  = df[test_mask]['bleaching_in_6wks']

print(f"Train: {len(X_train):} rows | Test: {len(X_test):} rows")
print(f"Bleaching events in train: {y_train.sum():} ({y_train.mean()*100:}%)")

# oversample bleaching events so model sees balanced classes
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Model Training
scale = (y_train == 0).sum() / y_train.sum()

model = XGBClassifier(
    n_estimators=10000,
    max_depth=30,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    min_child_weight=5,
    gamma=1,               # only split if it meaningfully reduces loss
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.5,        # L2 regularization
    random_state=1,
    n_jobs=-1
)

model.fit(X_train_sm, y_train_sm, eval_set=[(X_test, y_test)], verbose=100)

y_proba = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[f1_scores.argmax()]
print(f"\nOptimal threshold: {best_threshold:.3f}")

y_pred_tuned = (y_proba >= best_threshold).astype(int)
print("\n── Final Results ──")
print(classification_report(y_test, y_pred_tuned))
print(f"ROC-AUC:       {roc_auc_score(y_test, y_proba):.4f}")
print(f"Avg Precision: {average_precision_score(y_test, y_proba):.4f}")

cm   = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Bleaching', 'Bleaching'])
disp.plot(cmap='Blues')
plt.title('CoralGuard — Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# Saving model
with open("coralguard_model.pkl",    "wb") as f: pickle.dump(model,          f)
with open("coralguard_features.pkl", "wb") as f: pickle.dump(FEATURES,       f)
with open("coralguard_threshold.pkl","wb") as f: pickle.dump(best_threshold, f)
