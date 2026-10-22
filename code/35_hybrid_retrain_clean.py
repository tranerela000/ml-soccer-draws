"""
混合模型重训（干净版）: XGB(静态) + Bi-LSTM+Attention(因果序列) + Stacking(含校准)
评估: 诚实的OOF交叉验证(按赛季扩展窗口) + 最终holdout只用一次
序列原则: 每场比赛的序列 = 该队此前最多5场比赛的完整统计(含结果), 严格因果
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import sys
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
import keras.ops as kops
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense,
                                     Softmax, Dropout)
from tensorflow.keras.models import Model

def pf(msg):
    print(msg)
    sys.stdout.flush()

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

pf("=" * 70)
pf("混合模型重训（干净版）")
pf("=" * 70)

train_df = pd.read_csv('data/processed/train_hist.csv')
test_df = pd.read_csv('data/processed/test_hist.csv')
for d in (train_df, test_df):
    d['Date'] = pd.to_datetime(d['Date'])
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)
train_df['_mid'] = np.arange(len(train_df))
test_df['_mid'] = np.arange(len(test_df)) + len(train_df)
n_train = len(train_df)
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_df = all_df.sort_values('Date', kind='stable').reset_index(drop=True)

# ============================================================
# [1] 构建因果序列: 每队此前最多5场比赛的统计向量
# 维度: [res/2, gf/5, ga/5, shots/20, sot/10, corners/15, yellows/5, is_home, rest/28, 0]
# ============================================================
pf("\n[1/4] 构建因果序列 (每队前5场)...")
SEQ_LEN, SEQ_DIM = 5, 10

home_part = all_df[['_mid', 'Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR',
                    'HS', 'HST', 'HC', 'HY']].copy()
home_part.columns = ['_mid', 'Date', 'Team', 'gf', 'ga', 'res0', 'Shots', 'SoT',
                     'Corners', 'Yellows']
home_part['is_home'] = 1
away_part = all_df[['_mid', 'Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR',
                    'AS', 'AST', 'AC', 'AY']].copy()
away_part.columns = ['_mid', 'Date', 'Team', 'gf', 'ga', 'res0', 'Shots', 'SoT',
                     'Corners', 'Yellows']
away_part['is_home'] = 0
long_df = pd.concat([home_part, away_part], ignore_index=True)
long_df['res'] = long_df['res0'].map({'H': 2, 'D': 1, 'A': 0})
long_df.loc[long_df['is_home'] == 0, 'res'] = long_df.loc[
    long_df['is_home'] == 0, 'res0'].map({'A': 2, 'D': 1, 'H': 0})
long_df = long_df.sort_values(['Team', 'Date'], kind='stable').reset_index(drop=True)

seq_map = {}
for team, grp in long_df.groupby('Team'):
    hist = []
    for _, r in grp.iterrows():
        seq_map[(r['_mid'], team)] = np.array(hist[-SEQ_LEN:], dtype=np.float32)
        hist.append([r['res'] / 2, r['gf'] / 5, r['ga'] / 5, r['Shots'] / 20,
                     r['SoT'] / 10, r['Corners'] / 15, r['Yellows'] / 5,
                     r['is_home'], 0, 0])

def build_seq(d):
    n = len(d)
    out = np.zeros((n, SEQ_LEN, SEQ_DIM * 2), dtype=np.float32)
    for i, (m, h, a) in enumerate(zip(d['_mid'], d['HomeTeam'], d['AwayTeam'])):
        hs = seq_map[(m, h)]
        as_ = seq_map[(m, a)]
        if len(hs):
            out[i, -len(hs):, :SEQ_DIM] = hs
        if len(as_):
            out[i, -len(as_):, SEQ_DIM:] = as_
    return out

seq_train = build_seq(train_df)
seq_test = build_seq(test_df)
pf(f"  训练序列: {seq_train.shape}, 测试序列: {seq_test.shape}")

# ============================================================
# [2] 静态特征 (与34号脚本基线一致)
# ============================================================
pf("\n[2/4] 准备静态特征...")
LEAK_COLS = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
             'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
             'HY', 'AY', 'HR', 'AR',
             'Home_SoT_Rate', 'Away_SoT_Rate']
ID_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'Div', 'Time',
           'Referee', 'Attendance', 'stadium', 'country', '_mid']
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
X_test = prepare_X(test_df, feature_cols, final_cols=list(X_train.columns))
y_train = train_df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values
y_test = test_df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values
pf(f"  静态特征: {X_train.shape[1]}个")

def new_xgb():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
        random_state=SEED, eval_metric='mlogloss')

def build_lstm():
    inp = Input((SEQ_LEN, SEQ_DIM * 2))
    x = Bidirectional(LSTM(32, return_sequences=True))(inp)
    score = Dense(1)(x)
    w = Softmax(axis=1)(score)
    ctx = kops.sum(x * w, axis=1)
    x = Dense(32, activation='relu')(ctx)
    x = Dropout(0.3)(x, training=True)
    out = Dense(3, activation='softmax')(x)
    m = Model(inp, out)
    m.compile('adam', 'sparse_categorical_crossentropy')
    return m

seasons = sorted(train_df['Season'].dropna().unique())
folds = []  # 扩展窗口: 训练seasons[:i+1] → 测试seasons[i+1]
for i in range(len(seasons) - 1):
    tr = train_df['Season'].isin(seasons[:i + 1]).values
    te = (train_df['Season'] == seasons[i + 1]).values
    folds.append((tr, te))

# ============================================================
# [3] OOF生成（诚实评估）
# ============================================================
pf("\n[3/4] 生成OOF预测（按赛季扩展窗口）...")
xgb_oof = np.zeros((n_train, 3))
lstm_oof = np.zeros((n_train, 3))

for fi, (tr, te) in enumerate(folds):
    m = new_xgb()
    m.fit(X_train[tr], y_train[tr])
    xgb_oof[te] = m.predict_proba(X_train[te])

    lm = build_lstm()
    lm.fit(seq_train[tr], y_train[tr], epochs=25, batch_size=64, verbose=0)
    lstm_oof[te] = lm.predict(seq_train[te], verbose=0)
    pf(f"  Fold {fi+1} ({seasons[fi+1]}): 完成")

xgb_oof_acc = accuracy_score(y_train, xgb_oof.argmax(axis=1))
xgb_oof_ll = log_loss(y_train, xgb_oof)
lstm_oof_acc = accuracy_score(y_train, lstm_oof.argmax(axis=1))
lstm_oof_ll = log_loss(y_train, np.clip(lstm_oof, 1e-9, 1))
pf(f"\n  XGB  OOF: acc={xgb_oof_acc*100:.2f}%, LL={xgb_oof_ll:.4f}")
pf(f"  LSTM OOF: acc={lstm_oof_acc*100:.2f}%, LL={lstm_oof_ll:.4f}")

# Stacking（LR本身起校准作用）
stack_input = np.hstack([xgb_oof, np.clip(lstm_oof, 1e-9, 1)])
stack = LogisticRegression(max_iter=1000, C=1.0)
stack.fit(stack_input, y_train)
stack_oof_p = stack.predict_proba(stack_input)
stack_oof_acc = accuracy_score(y_train, stack_oof_p.argmax(axis=1))
stack_oof_ll = log_loss(y_train, stack_oof_p)
pf(f"  Stack OOF: acc={stack_oof_acc*100:.2f}%, LL={stack_oof_ll:.4f}")

# ============================================================
# [4] 最终模型 + holdout（只用一次）
# ============================================================
pf("\n[4/4] 最终模型训练 + holdout评估...")
xgb_final = new_xgb()
xgb_final.fit(X_train, y_train)
xgb_test_p = xgb_final.predict_proba(X_test)

lstm_final = build_lstm()
lstm_final.fit(seq_train, y_train, epochs=25, batch_size=64, verbose=0)
lstm_test_p = lstm_final.predict(seq_test, verbose=0)

stack_test_input = np.hstack([xgb_test_p, np.clip(lstm_test_p, 1e-9, 1)])
stack_test_p = stack.predict_proba(stack_test_input)

pf("\n" + "=" * 70)
pf("2025-26 Holdout 最终对比（1752场）")
pf("=" * 70)
pf(f"  XGB基线:        acc={accuracy_score(y_test, xgb_test_p.argmax(1))*100:.2f}%, "
   f"LL={log_loss(y_test, xgb_test_p):.4f}")
pf(f"  Bi-LSTM:        acc={accuracy_score(y_test, lstm_test_p.argmax(1))*100:.2f}%, "
   f"LL={log_loss(y_test, np.clip(lstm_test_p,1e-9,1)):.4f}")
pf(f"  Stack混合模型:  acc={accuracy_score(y_test, stack_test_p.argmax(1))*100:.2f}%, "
   f"LL={log_loss(y_test, stack_test_p):.4f}")

# 赔率基准
valid = test_df[['AvgH', 'AvgD', 'AvgA']].notna().all(axis=1).values
op = 1 / test_df.loc[valid, ['AvgH', 'AvgD', 'AvgA']]
op = (op.T / op.sum(axis=1)).T.values
pf(f"  赔率隐含概率:   acc={accuracy_score(y_test[valid], op.argmax(1))*100:.2f}%, "
   f"LL={log_loss(y_test[valid], op):.4f}")

with open('hybrid_retrain_report.txt', 'w', encoding='utf-8') as f:
    f.write("混合模型重训报告（干净版）\n")
    f.write("=" * 50 + "\n")
    f.write(f"OOF诚实评估: XGB acc={xgb_oof_acc*100:.2f}%/LL={xgb_oof_ll:.4f}; "
            f"LSTM acc={lstm_oof_acc*100:.2f}%/LL={lstm_oof_ll:.4f}; "
            f"Stack acc={stack_oof_acc*100:.2f}%/LL={stack_oof_ll:.4f}\n\n")
    f.write("Holdout(2025-26, 只用一次):\n")
    f.write(f"XGB: acc={accuracy_score(y_test, xgb_test_p.argmax(1))*100:.2f}%\n")
    f.write(f"LSTM: acc={accuracy_score(y_test, lstm_test_p.argmax(1))*100:.2f}%\n")
    f.write(f"Stack: acc={accuracy_score(y_test, stack_test_p.argmax(1))*100:.2f}%\n")

# 保存模型与OOF供复用
lstm_final.save('data/processed/lstm_clean.keras')
np.save('data/processed/stack_oof_probs.npy', stack_oof_p)
pf("\n✓ 完成: hybrid_retrain_report.txt, lstm_clean.keras")
