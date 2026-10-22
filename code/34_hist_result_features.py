"""
历史结果型数据滚动特征（射门/射正/角球/牌数）+ 消融验证
原则：只用"过去比赛"的统计（shift(1)后取滚动均值），当场比赛信息绝不入特征
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
pf("历史结果型数据滚动特征 + 消融验证")
pf("=" * 70)

train_df = pd.read_csv('data/processed/train_strict.csv')
test_df = pd.read_csv('data/processed/test_strict.csv')
for d in (train_df, test_df):
    d['Date'] = pd.to_datetime(d['Date'])
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)
train_df['_mid'] = np.arange(len(train_df))
test_df['_mid'] = np.arange(len(test_df)) + len(train_df)

all_df = pd.concat([train_df, test_df], ignore_index=True)
all_df = all_df.sort_values('Date', kind='stable').reset_index(drop=True)

# ============================================================
# 每支球队的历史滚动统计（L5，严格因果：shift(1)后rolling）
# ============================================================
pf("\n[1/2] 计算历史射门/射正/角球/牌数滚动特征（L5）...")

long_df = pd.concat([
    all_df[['_mid', 'Date', 'HomeTeam', 'HS', 'HST', 'HC', 'HY']]
        .rename(columns={'HomeTeam': 'Team', 'HS': 'Shots', 'HST': 'SoT',
                         'HC': 'Corners', 'HY': 'Yellows'})
        .assign(_is_home=1),
    all_df[['_mid', 'Date', 'AwayTeam', 'AS', 'AST', 'AC', 'AY']]
        .rename(columns={'AwayTeam': 'Team', 'AS': 'Shots', 'AST': 'SoT',
                         'AC': 'Corners', 'AY': 'Yellows'})
        .assign(_is_home=0)
], ignore_index=True).sort_values(['Team', 'Date'], kind='stable').reset_index(drop=True)

g = long_df.groupby('Team')
for col in ['Shots', 'SoT', 'Corners', 'Yellows']:
    # 关键：shift(1)确保只用此前比赛
    long_df[f'{col}_L5'] = g[col].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())

feat_map = {}
for _, r in long_df.iterrows():
    feat_map[(r['_mid'], r['Team'])] = (r['Shots_L5'], r['SoT_L5'],
                                        r['Corners_L5'], r['Yellows_L5'])

hist_cols = []
for d in (train_df, test_df):
    d['Home_L5_Shots'] = [feat_map[(m, h)][0] for m, h in zip(d['_mid'], d['HomeTeam'])]
    d['Away_L5_Shots'] = [feat_map[(m, a)][0] for m, a in zip(d['_mid'], d['AwayTeam'])]
    d['Home_L5_SoT'] = [feat_map[(m, h)][1] for m, h in zip(d['_mid'], d['HomeTeam'])]
    d['Away_L5_SoT'] = [feat_map[(m, a)][1] for m, a in zip(d['_mid'], d['AwayTeam'])]
    d['Home_L5_Corners'] = [feat_map[(m, h)][2] for m, h in zip(d['_mid'], d['HomeTeam'])]
    d['Away_L5_Corners'] = [feat_map[(m, a)][2] for m, a in zip(d['_mid'], d['AwayTeam'])]
    d['Home_L5_Yellows'] = [feat_map[(m, h)][3] for m, h in zip(d['_mid'], d['HomeTeam'])]
    d['Away_L5_Yellows'] = [feat_map[(m, a)][3] for m, a in zip(d['_mid'], d['AwayTeam'])]
    # 差值特征
    d['Shots_Diff'] = d['Home_L5_Shots'] - d['Away_L5_Shots']
    d['SoT_Diff'] = d['Home_L5_SoT'] - d['Away_L5_SoT']
    d['Corners_Diff'] = d['Home_L5_Corners'] - d['Away_L5_Corners']
    # 缺失（赛季初无历史）用联盟均值近似
    for c in ['Home_L5_Shots', 'Away_L5_Shots', 'Home_L5_SoT', 'Away_L5_SoT',
              'Home_L5_Corners', 'Away_L5_Corners', 'Home_L5_Yellows', 'Away_L5_Yellows']:
        d[c] = d[c].fillna(d[c].mean())

hist_cols = ['Home_L5_Shots', 'Away_L5_Shots', 'Home_L5_SoT', 'Away_L5_SoT',
             'Home_L5_Corners', 'Away_L5_Corners', 'Home_L5_Yellows', 'Away_L5_Yellows',
             'Shots_Diff', 'SoT_Diff', 'Corners_Diff']
pf(f"  新增特征: {len(hist_cols)}个")
pf(f"  示例(主队L5射门): 均值{train_df['Home_L5_Shots'].mean():.2f}, "
   f"缺失填充后无NaN: {train_df['Home_L5_Shots'].notna().all()}")

# 保存
train_df.to_csv('data/processed/train_hist.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('data/processed/test_hist.csv', index=False, encoding='utf-8-sig')
pf("  已保存: train_hist.csv / test_hist.csv")

# ============================================================
# 消融验证
# ============================================================
pf("\n[2/2] 消融验证...")

LEAK_COLS = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
             'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
             'HY', 'AY', 'HR', 'AR',
             'Home_SoT_Rate', 'Away_SoT_Rate']
ID_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'Div', 'Time',
           'Referee', 'Attendance', 'stadium', 'country', '_mid']

feature_cols = [c for c in train_df.columns
                if c not in LEAK_COLS and c not in ID_COLS and c != 'FTR']
feature_cols = [c for c in feature_cols if c in test_df.columns]
feature_cols = [c for c in feature_cols if c not in set(hist_cols)]  # 基线剔除新特征
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

y_all = train_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
y_test_final = test_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

def new_model():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
        random_state=42, eval_metric='mlogloss')

seasons = sorted(train_df['Season'].dropna().unique())

def evaluate(cols, name):
    X = prepare_X(train_df, cols)
    X_te = prepare_X(test_df, cols, final_cols=list(X.columns))
    cv_accs, cv_lls = [], []
    for i in range(len(seasons) - 1):
        tr = train_df['Season'].isin(seasons[:i+1]).values
        te = (train_df['Season'] == seasons[i+1]).values
        m = new_model()
        m.fit(X[tr], y_all[tr])
        p = m.predict_proba(X[te])
        cv_accs.append(accuracy_score(y_all[te], p.argmax(axis=1)))
        cv_lls.append(log_loss(y_all[te], p))
    m = new_model()
    m.fit(X, y_all)
    p_te = m.predict_proba(X_te)
    hold = accuracy_score(y_test_final, p_te.argmax(axis=1))
    hold_ll = log_loss(y_test_final, p_te)
    pf(f"  {name}: CV={np.mean(cv_accs)*100:.2f}%, CV_LL={np.mean(cv_lls):.4f}, "
       f"Holdout={hold*100:.2f}% (LL {hold_ll:.4f})")
    return np.mean(cv_accs), np.mean(cv_lls), hold

X_base = evaluate(feature_cols, '基线(strict特征)')
X_hist = evaluate(feature_cols + hist_cols, '基线+历史统计滚动')
pf(f"\n边际变化: CV{(X_hist[0]-X_base[0])*100:+.2f}%, "
   f"LL {X_hist[1]-X_base[1]:+.4f}, Holdout{(X_hist[2]-X_base[2])*100:+.2f}%")

with open('hist_features_report.txt', 'w', encoding='utf-8') as f:
    f.write("历史结果型数据滚动特征消融报告\n")
    f.write(f"基线: CV={X_base[0]*100:.2f}%, LL={X_base[1]:.4f}, Holdout={X_base[2]*100:.2f}%\n")
    f.write(f"+历史统计: CV={X_hist[0]*100:.2f}%, LL={X_hist[1]:.4f}, "
            f"Holdout={X_hist[2]*100:.2f}%\n")
    f.write(f"决策标准: Holdout>=+0.5%且CV不降 → 保留\n")
pf("\n✓ 完成: hist_features_report.txt")
