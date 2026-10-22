"""
模型 vs 博彩公司赔率 基准差距分析
在同一测试集(2025-26)上对比:
1. 准确率: 模型预测 vs 赔率选热门(最低赔率)
2. log loss: 模型概率 vs 赔率隐含概率(校准基准)
"""
import pandas as pd
import numpy as np
import sys
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

def pf(msg):
    print(msg)
    sys.stdout.flush()

pf("=" * 70)
pf("模型 vs 博彩公司赔率 差距分析（2025-26赛季，1752场）")
pf("=" * 70)

train_df = pd.read_csv('data/processed/train_strict.csv')
test_df = pd.read_csv('data/processed/test_strict.csv')
for d in (train_df, test_df):
    d['Date'] = pd.to_datetime(d['Date'])
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)

# ---------- 特征准备（与31号脚本一致） ----------
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

X_all = prepare_X(train_df, feature_cols)
y_all = train_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
X_test = prepare_X(test_df, feature_cols, final_cols=list(X_all.columns))
y_test = test_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# ---------- 模型预测概率 ----------
pf("\n训练模型...")
model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
    random_state=42, eval_metric='mlogloss')
model.fit(X_all, y_all)
model_probs = model.predict_proba(X_test)   # 列顺序按标签0,1,2 = H,D,A
model_pred = model_probs.argmax(axis=1)
model_acc = accuracy_score(y_test, model_pred)

# ---------- 赔率基准 ----------
def odds_probs(d, prefix):
    p = 1 / d[[f'{prefix}H', f'{prefix}D', f'{prefix}A']]
    return (p.T / p.sum(axis=1)).T.values

def odds_favorite(d, prefix):
    return np.argmin(d[[f'{prefix}H', f'{prefix}D', f'{prefix}A']].values, axis=1)

pf("\n" + "=" * 70)
pf(f"{'基准':<22}{'准确率':>10}{'Log Loss':>12}")
pf("=" * 70)
pf(f"{'我们的模型':<22}{model_acc*100:>9.2f}%{log_loss(y_test, model_probs):>12.4f}")

for prefix, name in [('Avg', '平均赔率(Avg)'), ('B365', 'Bet365'),
                     ('PS', 'Pinnacle'), ('BW', 'Bet&Win'), ('Max', '最高赔率(Max)')]:
    if f'{prefix}H' in test_df.columns:
        valid = test_df[[f'{prefix}H', f'{prefix}D', f'{prefix}A']].notna().all(axis=1).values
        acc = accuracy_score(y_test[valid], odds_favorite(test_df[valid], prefix))
        ll = log_loss(y_test[valid], odds_probs(test_df[valid], prefix))
        pf(f"{name:<22}{acc*100:>9.2f}%{ll:>12.4f}  (有效{valid.sum()}场)")

# ---------- 差距解读 ----------
pf("\n" + "=" * 70)
pf("差距解读")
pf("=" * 70)
avg_acc = accuracy_score(y_test, odds_favorite(test_df, 'Avg'))
diff = (model_acc - avg_acc) * 100
pf(f"模型 vs 平均赔率热门: {diff:+.2f} 个百分点")
if diff >= 1:
    pf("→ 模型显著优于市场基准")
elif diff >= -0.5:
    pf("→ 模型与市场基准基本持平（±0.5%以内即无实质差距）")
else:
    pf("→ 模型落后于市场基准")

# 分结果看模型 vs 赔率的选择差异
pf("\n模型与赔率热门选择不一致的比赛:")
disagree = model_pred != odds_favorite(test_df, 'Avg')
if disagree.sum() > 0:
    sub_acc = accuracy_score(y_test[disagree], model_pred[disagree])
    fav_acc = accuracy_score(y_test[disagree], odds_favorite(test_df, 'Avg')[disagree])
    pf(f"  分歧比赛数: {disagree.sum()} ({disagree.mean()*100:.1f}%)")
    pf(f"  分歧比赛中 模型正确率: {sub_acc*100:.2f}%")
    pf(f"  分歧比赛中 赔率热门正确率: {fav_acc*100:.2f}%")
