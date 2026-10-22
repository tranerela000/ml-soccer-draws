"""
Tier-1特征补充与边际价值验证
1. 特征组A: 休息天数/赛程密度（从现有日期严格因果计算）
2. 特征组B: 赔率衍生特征（隐含概率、庄家分歧、开盘→收盘变化）
3. 消融验证: 基线 / +休息 / +赔率 / +全部，各自测CV与holdout边际价值
"""
import pandas as pd
import numpy as np
import sys
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def pf(msg):
    print(msg)
    sys.stdout.flush()

pf("=" * 70)
pf("Tier-1特征补充与验证")
pf("=" * 70)

# 加载严格分离的数据
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
pf(f"合并: {len(all_df)} 场 (train {len(train_df)}, test {len(test_df)})")
assert train_df['Date'].max() < test_df['Date'].min(), "时间重叠！"

# ============================================================
# 特征组A: 休息天数/赛程密度（严格因果：shift(1)只看此前比赛）
# ============================================================
pf("\n[A] 计算休息天数/赛程密度（严格因果）...")

long_df = pd.concat([
    all_df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'Team'}).assign(_mid=all_df.index),
    all_df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'Team'}).assign(_mid=all_df.index)
], ignore_index=True).sort_values(['Team', 'Date']).reset_index(drop=True)

rest_map = {}
cong_map = {}
for team, g in long_df.groupby('Team'):
    dates = g['Date'].values
    mids = g['_mid'].values
    for i in range(len(dates)):
        if i == 0:
            rest_map[(mids[i], team)] = np.nan
            cong_map[(mids[i], team)] = 0
        else:
            rest_map[(mids[i], team)] = (dates[i] - dates[i-1]) / np.timedelta64(1, 'D')
            # 近14天比赛数（不含当前）
            j = i
            while j > 0 and (dates[i] - dates[j-1]) <= np.timedelta64(14, 'D'):
                j -= 1
            cong_map[(mids[i], team)] = i - j

for d in (train_df, test_df):
    d['Home_RestDays'] = [rest_map[(m, h)] for m, h in zip(d['_mid'], d['HomeTeam'])]
    d['Away_RestDays'] = [rest_map[(m, a)] for m, a in zip(d['_mid'], d['AwayTeam'])]
    d['RestDays_Diff'] = d['Home_RestDays'] - d['Away_RestDays']
    d['Home_Matches14d'] = [cong_map[(m, h)] for m, h in zip(d['_mid'], d['HomeTeam'])]
    d['Away_Matches14d'] = [cong_map[(m, a)] for m, a in zip(d['_mid'], d['AwayTeam'])]
    d['Congestion_Diff'] = d['Home_Matches14d'] - d['Away_Matches14d']
    d['Home_RestDays'] = d['Home_RestDays'].fillna(7)  # 首场按标准周赛程处理
    d['Away_RestDays'] = d['Away_RestDays'].fillna(7)
    d['RestDays_Diff'] = d['RestDays_Diff'].fillna(0)
    # 长期缺席（升班马等）封顶28天，避免极端值噪声
    d['Home_RestDays'] = d['Home_RestDays'].clip(2, 28)
    d['Away_RestDays'] = d['Away_RestDays'].clip(2, 28)
    d['RestDays_Diff'] = d['RestDays_Diff'].clip(-26, 26)

pf(f"  休息天数分布: 均值{train_df['Home_RestDays'].mean():.1f}天, "
   f"中位数{train_df['Home_RestDays'].median():.0f}天, "
   f"范围{train_df['Home_RestDays'].min():.0f}~{train_df['Home_RestDays'].max():.0f}天")

rest_cols = ['Home_RestDays', 'Away_RestDays', 'RestDays_Diff',
             'Home_Matches14d', 'Away_Matches14d', 'Congestion_Diff']

# ============================================================
# 特征组B: 赔率衍生特征（赛前静态信息，无因果问题）
# ============================================================
pf("\n[B] 计算赔率衍生特征...")

# 找训练测试都有的庄家
books = [b for b in ['B365', 'BW', 'IW', 'PS', 'WH', 'VC']
         if f'{b}H' in train_df.columns and f'{b}H' in test_df.columns]
pf(f"  可用庄家: {books}")

for d in (train_df, test_df):
    # B1: 平均收盘赔率的归一化隐含概率
    inv = pd.concat([1 / d['AvgH'], 1 / d['AvgD'], 1 / d['AvgA']], axis=1)
    inv.columns = ['_H', '_D', '_A']
    tot = inv.sum(axis=1)
    d['Prob_H'] = inv['_H'] / tot
    d['Prob_D'] = inv['_D'] / tot
    d['Prob_A'] = inv['_A'] / tot

    # B2: 庄家分歧度（各庄家主胜赔率的标准差）
    d['Odds_H_Std'] = d[[f'{b}H' for b in books]].std(axis=1).fillna(0)
    d['Odds_D_Std'] = d[[f'{b}D' for b in books]].std(axis=1).fillna(0)
    d['Odds_A_Std'] = d[[f'{b}A' for b in books]].std(axis=1).fillna(0)

    # B3: 开盘→收盘赔率变化（相对变化率）
    for src in ['B365', 'Avg']:
        for o in ['H', 'D', 'A']:
            d[f'Move_{src}_{o}'] = ((d[f'{src}C{o}'] - d[f'{src}{o}'])
                                    / d[f'{src}{o}']).fillna(0)

odds_cols = ['Prob_H', 'Prob_D', 'Prob_A',
             'Odds_H_Std', 'Odds_D_Std', 'Odds_A_Std',
             'Move_B365_H', 'Move_B365_D', 'Move_B365_A',
             'Move_Avg_H', 'Move_Avg_D', 'Move_Avg_A']

pf(f"  特征组A: {len(rest_cols)}个, 特征组B: {len(odds_cols)}个")

# ============================================================
# 基线特征（与31号脚本一致，排除全部泄漏列）
# ============================================================
LEAK_COLS = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
             'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
             'HY', 'AY', 'HR', 'AR',
             'Home_SoT_Rate', 'Away_SoT_Rate']
ID_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'Div', 'Time',
           'Referee', 'Attendance', 'stadium', 'country', '_mid']

feature_cols = [c for c in train_df.columns
                if c not in LEAK_COLS and c not in ID_COLS and c != 'FTR']
feature_cols = [c for c in feature_cols if c in test_df.columns]
# 关键修复：把Tier-1特征从基线中剔除，否则"基线"也包含新特征
tier1_all = set(rest_cols) | set(odds_cols)
feature_cols = [c for c in feature_cols if c not in tier1_all]

# 剔除缺失>50%
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

def make_config(rest=False, odds=False):
    cols = list(feature_cols)
    if rest:
        cols += rest_cols
    if odds:
        cols += odds_cols
    X = prepare_X(train_df, cols)
    X_te = prepare_X(test_df, cols, final_cols=list(X.columns))
    return X, X_te

# 干净基线模型（强正则化配置）
def new_model():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
        random_state=42, eval_metric='mlogloss')

seasons = sorted(train_df['Season'].dropna().unique())

def evaluate(X, X_te, name):
    """返回: CV各折测试准确率, holdout准确率"""
    cv_accs = []
    for i in range(len(seasons) - 1):
        tr_mask = train_df['Season'].isin(seasons[:i+1]).values
        te_mask = (train_df['Season'] == seasons[i+1]).values
        m = new_model()
        m.fit(X[tr_mask], y_all[tr_mask])
        cv_accs.append(accuracy_score(y_all[te_mask], m.predict(X[te_mask])))

    m = new_model()
    m.fit(X, y_all)
    hold = accuracy_score(y_test_final, m.predict(X_te))
    pf(f"  {name}: CV={np.mean(cv_accs)*100:.2f}% (各折{[f'{a*100:.1f}' for a in cv_accs]}), "
       f"Holdout={hold*100:.2f}%")
    return np.mean(cv_accs), hold

# ============================================================
# 消融验证
# ============================================================
pf("\n" + "=" * 70)
pf("消融验证（配置: XGB 100树/深度3/强正则化）")
pf("=" * 70)

results = {}
for name, rest, odds in [
    ('基线', False, False),
    ('基线+休息天数', True, False),
    ('基线+赔率衍生', False, True),
    ('基线+全部Tier1', True, True)
]:
    X, X_te = make_config(rest, odds)
    results[name] = evaluate(X, X_te, name)

# ============================================================
# 结论与决策
# ============================================================
pf("\n" + "=" * 70)
pf("边际价值分析")
pf("=" * 70)

base_cv, base_hold = results['基线']
for name in ['基线+休息天数', '基线+赔率衍生', '基线+全部Tier1']:
    cv, hold = results[name]
    dcv = (cv - base_cv) * 100
    dhold = (hold - base_hold) * 100
    pf(f"  {name}: CV变化{dcv:+.2f}%, Holdout变化{dhold:+.2f}%")

# 保存Tier-1数据（无论是否保留，供后续使用）
train_df.to_csv('data/processed/train_t1.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('data/processed/test_t1.csv', index=False, encoding='utf-8-sig')

# 检查Tier-1特征是否被模型实际使用
pf("\n" + "=" * 70)
pf("Tier-1特征使用情况（全部Tier1模型）")
pf("=" * 70)
X_full, _ = make_config(True, True)
m = new_model()
m.fit(X_full, y_all)
imp = pd.DataFrame({'f': X_full.columns, 'v': m.feature_importances_})
t1_imp = imp[imp['f'].isin([sanitize(c) for c in tier1_all])].sort_values('v', ascending=False)
if len(t1_imp) == 0:
    pf("  Tier-1特征全部未被使用（重要性为0）")
else:
    for _, row in t1_imp.iterrows():
        pf(f"  {row['f']}: {row['v']*100:.2f}%")

with open('tier1_validation_report.txt', 'w', encoding='utf-8') as f:
    f.write("Tier-1特征消融验证报告\n")
    f.write(f"模型: XGB 100树/深度3/强正则化\n\n")
    base_cv, base_hold = results['基线']
    f.write(f"基线: CV={base_cv*100:.2f}%, Holdout={base_hold*100:.2f}%\n")
    for name in ['基线+休息天数', '基线+赔率衍生', '基线+全部Tier1']:
        cv, hold = results[name]
        f.write(f"{name}: CV={cv*100:.2f}%({(cv-base_cv)*100:+.2f}%), "
                f"Holdout={hold*100:.2f}%({(hold-base_hold)*100:+.2f}%)\n")
    f.write("\n决策标准: Holdout提升>=0.5%且CV不下降 → 保留该特征组\n")

pf("\n✓ 已保存: data/processed/train_t1.csv, test_t1.csv, tier1_validation_report.txt")
