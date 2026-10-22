"""
McNemar检验：参考模型 vs 博彩赔率基准的配对正确率比较
回答4.1节"statistically indistinguishable"是否有统计依据
"""
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from xgboost import XGBClassifier


def pf(msg):
    print(msg)
    sys.stdout.flush()


train_df = pd.read_csv('data/processed/train_strict.csv')
test_df = pd.read_csv('data/processed/test_strict.csv')
for d in (train_df, test_df):
    d['Date'] = pd.to_datetime(d['Date'])
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)

LEAK_COLS = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
             'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
             'HY', 'AY', 'HR', 'AR',
             'Home_SoT_Rate', 'Away_SoT_Rate']
ID_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'Div', 'Time',
           'Referee', 'Attendance', 'stadium', 'country']

feature_cols = [c for c in train_df.columns
                if c not in LEAK_COLS and c not in ID_COLS and c != 'FTR']
feature_cols = [c for c in feature_cols if c in test_df.columns]
missing_ratio = train_df[feature_cols].isnull().mean()
feature_cols = [c for c in feature_cols if missing_ratio[c] <= 0.5]


def sanitize(name):
    return re.sub(r'[\[\]<>, ]', '_', str(name))


def prepare_X(df, cols, final_cols=None):
    X = df[cols].copy()
    X.columns = [sanitize(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    if final_cols is not None:
        X = X.reindex(columns=final_cols, fill_value=0)
    return X


X_train = prepare_X(train_df, feature_cols)
y_train = train_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
X_test = prepare_X(test_df, feature_cols, final_cols=list(X_train.columns))
y_test = test_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
    random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 只在赔率齐全的场次上配对比较（与赔率基准口径一致）
valid = test_df[['AvgH', 'AvgD', 'AvgA']].notna().all(axis=1).values
odds_pred = np.argmin(test_df[['AvgH', 'AvgD', 'AvgA']].values, axis=1)

y_v = y_test[valid]
m_correct = (pred[valid] == y_v)
o_correct = (odds_pred == y_v)

n = valid.sum()
b = int((m_correct & ~o_correct).sum())  # 模型对、赔率错
c = int((~m_correct & o_correct).sum())  # 模型错、赔率对
pf(f"配对样本: {n}场（赔率齐全）")
pf(f"模型准确率: {m_correct.mean()*100:.2f}% | 赔率基准: {o_correct.mean()*100:.2f}%")
pf(f"不一致格: 模型对&赔率错 b={b}, 模型错&赔率对 c={c}")
res = binomtest(min(b, c), b + c, 0.5)
pf(f"McNemar精确检验 p = {res.pvalue:.3f}")
pf(f"结论: {'无显著差异（支持indistinguishable表述）' if res.pvalue > 0.05 else '存在显著差异（需修改4.1措辞）'}")
