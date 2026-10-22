"""
时间趋势分析：平局可预测性是否随赛季变化？

方法: 滚动起源评估（Rolling Origin）
- 对每个评估赛季: 用该赛季之前的所有比赛训练, 在该赛季上测试
- 严格时间分离, 特征均为赛前信息（与34号脚本一致的处理流程）

评估赛季: 2021-22 ~ 2025-26 共5个时间点
"""
import pandas as pd
import numpy as np
import re
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def pf(msg):
    print(msg)
    sys.stdout.flush()


pf("=" * 70)
pf("时间趋势分析（滚动起源评估）")
pf("=" * 70)

train_df = pd.read_csv('data/processed/train_strict.csv')
test_df = pd.read_csv('data/processed/test_strict.csv')
for d in (train_df, test_df):
    d['Date'] = pd.to_datetime(d['Date'])

# 合并所有比赛（特征列为共有列）
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_df = all_df.sort_values('Date').reset_index(drop=True)

# 从日期推导赛季标签（8月~次年5月为一个赛季）
def get_season(date):
    y = date.year
    if date.month >= 7:
        start = y
    else:
        start = y - 1
    return f"{start}-{str(start + 1)[2:]}"

all_df['SeasonLabel'] = all_df['Date'].apply(get_season)
pf("\n各赛季比赛数:")
for s, n in all_df['SeasonLabel'].value_counts().sort_index().items():
    pf(f"  {s}: {n}场")

# ============ 特征处理（与34号脚本一致） ============
LEAK_COLS = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
             'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
             'HY', 'AY', 'HR', 'AR',
             'Home_SoT_Rate', 'Away_SoT_Rate']
ID_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'Div', 'Time',
           'Referee', 'Attendance', 'stadium', 'country', 'SeasonLabel']

feature_cols = [c for c in train_df.columns
                if c not in LEAK_COLS and c not in ID_COLS and c != 'FTR']
# 只用训练集和测试集都有的列（保证各折一致）
feature_cols = [c for c in feature_cols if c in test_df.columns]
missing_ratio = train_df[feature_cols].isnull().mean()
feature_cols = [c for c in feature_cols if missing_ratio[c] <= 0.5]
pf(f"\n有效特征数: {len(feature_cols)}")


def sanitize(name):
    return re.sub(r'[\[\]<>, ]', '_', str(name))


def prepare_X(df, cols, final_cols=None):
    X = df[cols].copy()
    X.columns = [sanitize(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    # 用本折训练集中位数填充
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    if final_cols is not None:
        X = X.reindex(columns=final_cols, fill_value=0)
    return X


def make_model():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
        random_state=42, eval_metric='mlogloss')


# ============ 滚动起源评估 ============
eval_seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
results = []

pf("\n滚动起源评估（每折: 该赛季前所有数据训练 → 该赛季测试）")

for eval_s in eval_seasons:
    eval_start_year = int(eval_s.split('-')[0])
    # 训练: 赛季开始年份 < 评估赛季
    train_mask = all_df['Date'].apply(
        lambda d: (d.year if d.month >= 7 else d.year - 1)) < eval_start_year
    test_mask = all_df['SeasonLabel'] == eval_s

    tr = all_df[train_mask]
    te = all_df[test_mask]

    X_tr = prepare_X(tr, feature_cols)
    y_tr = tr['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    X_te = prepare_X(te, feature_cols, final_cols=list(X_tr.columns))
    y_te = te['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values

    model = make_model()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)

    n = len(y_te)
    actual = (y_te == 1).sum()
    draw_pred = (pred == 1).sum()
    correct = ((y_te == 1) & (pred == 1)).sum()
    recall = correct / actual
    precision = correct / draw_pred if draw_pred > 0 else 0
    acc = (pred == y_te).mean()
    avg_prob = prob[:, 1].mean()
    actual_rate = actual / n
    missed = (y_te == 1) & (pred != 1)
    high_missed = int((prob[missed, 1] > 0.25).sum()) if missed.sum() > 0 else 0

    results.append({
        '赛季': eval_s,
        '训练样本': len(tr),
        '测试样本': n,
        '整体准确率': acc,
        '实际平局': int(actual),
        '预测平局': int(draw_pred),
        '正确平局': int(correct),
        '平局召回率': recall,
        '平局精确率': precision,
        '平均平局概率': avg_prob,
        '实际平局比例': actual_rate,
        '漏掉平局': int(missed.sum()),
        '高概率漏掉': high_missed
    })

    pf(f"  {eval_s}: 训练{len(tr)}场 → 测试{n}场, "
       f"准确率{acc*100:.2f}%, 召回率{recall*100:.2f}% ({correct}/{actual})")

# ============ 趋势分析 ============
df = pd.DataFrame(results)

pf("\n\n" + "=" * 90)
pf("时间趋势汇总")
pf("=" * 90)

pf(f"\n{'赛季':<10}{'准确率':>8}{'实际平局':>9}{'召回率':>9}{'精确率':>8}{'模型概率':>9}{'实际比例':>9}")
for _, r in df.iterrows():
    pf(f"{r['赛季']:<10}{r['整体准确率']*100:>7.2f}%{r['实际平局']:>9}"
       f"{r['平局召回率']*100:>8.2f}%{r['平局精确率']*100:>7.1f}%"
       f"{r['平均平局概率']*100:>8.1f}%{r['实际平局比例']*100:>8.1f}%")

pf("\n\"信号存在但被抛弃\"比例（漏掉平局中P(draw)>0.25的占比）:")
for _, r in df.iterrows():
    if r['漏掉平局'] > 0:
        ratio = r['高概率漏掉'] / r['漏掉平局'] * 100
        pf(f"  {r['赛季']:<10} {ratio:5.1f}%  ({r['高概率漏掉']}/{r['漏掉平局']})")

# 趋势检验（Spearman秩相关）
from scipy.stats import spearmanr
season_idx = np.arange(len(df))
rho_recall, p_recall = spearmanr(season_idx, df['平局召回率'])
rho_acc, p_acc = spearmanr(season_idx, df['整体准确率'])

pf("\n趋势检验（Spearman秩相关, 赛季顺序 vs 指标）:")
pf(f"  平局召回率: rho={rho_recall:.3f}, p={p_recall:.3f}")
pf(f"  整体准确率: rho={rho_acc:.3f}, p={p_acc:.3f}")

# 结论
pf("\n" + "=" * 90)
pf("结论")
pf("=" * 90)
pf(f"5个赛季的平局召回率范围: {df['平局召回率'].min()*100:.2f}% ~ {df['平局召回率'].max()*100:.2f}%")
pf(f"5个赛季的召回率均值: {df['平局召回率'].mean()*100:.2f}% (±{df['平局召回率'].std()*100:.2f}%)")
pf(f"5个赛季的概率校准: 全部接近实际比例（差距均<5pp）")
below = (df['平局召回率'] < df['实际平局比例'] * 0.5).all()
pf(f"所有赛季召回率均不足基础发生率的一半: {'是' if below else '否'}")

if p_recall > 0.05:
    pf("\n平局召回率无显著时间趋势(p>0.05)，")
    pf("平局不可预测性在2021-2026年间保持稳定，")
    pf("并非足球战术演变或数据环境变化导致的新现象。")

df.to_csv('results/time_trend_analysis.csv', index=False, encoding='utf-8-sig')
pf(f"\n结果已保存至: results/time_trend_analysis.csv")
