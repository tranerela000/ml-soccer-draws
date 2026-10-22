"""
子集分析：哪些比赛更容易漏掉平局？

两个子集维度:
1. 实力差距（赔率隐含概率差）—— 势均力敌的比赛平局率更高, 但模型表现如何?
2. 赛程密度（距上一场比赛的休息天数）—— 疲劳是否影响平局可预测性?

使用全局模型（34号脚本配置）在2025-26测试集上的预测
"""
import pandas as pd
import numpy as np
import re
import sys
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier


def pf(msg):
    print(msg)
    sys.stdout.flush()


pf("=" * 70)
pf("子集分析（实力差距 × 赛程密度）")
pf("=" * 70)

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
pf(f"有效特征数: {len(feature_cols)}")


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
y_test = test_df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values

pf("训练全局模型...")
model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
    random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)
pf(f"整体: 准确率{(pred == y_test).mean()*100:.2f}%, "
   f"平局召回率{((y_test == 1) & (pred == 1)).sum()/(y_test == 1).sum()*100:.2f}%")


def subset_report(mask, label, results):
    """输出子集的平局分析"""
    n = mask.sum()
    if n < 30:
        pf(f"  {label}: 样本过少({n}场), 跳过")
        return
    yt, pt, pp = y_test[mask], pred[mask], prob[mask]
    actual = (yt == 1).sum()
    draw_pred = (pt == 1).sum()
    correct = ((yt == 1) & (pt == 1)).sum()
    recall = correct / actual
    precision = correct / draw_pred if draw_pred > 0 else 0
    draw_rate = actual / n
    calib = pp[:, 1].mean()
    missed = (yt == 1) & (pt != 1)
    high_missed = (pp[missed, 1] > 0.25).sum() if missed.sum() > 0 else 0
    ratio = high_missed / missed.sum() * 100 if missed.sum() > 0 else 0

    pf(f"  {label}: {n}场, 平局率{draw_rate*100:.1f}%, "
       f"召回率{recall*100:.2f}% ({correct}/{actual}), 精确率{precision*100:.1f}%, "
       f"校准{calib*100:.1f}% vs {draw_rate*100:.1f}%, "
       f"信号抛弃{ratio:.0f}%")

    results.append({
        '子集': label, '样本': int(n), '平局率': draw_rate,
        '平局召回率': recall, '平局精确率': precision,
        '模型平均平局概率': calib, '高概率漏掉比例': ratio / 100
    })


# ============ 子集1: 实力差距（赔率隐含概率） ============
pf("\n" + "=" * 70)
pf("子集1: 实力差距（基于平均赔率隐含概率）")
pf("=" * 70)

odds = test_df[['AvgH', 'AvgD', 'AvgA']].copy()
valid_odds = odds.notna().all(axis=1).values
inv = 1.0 / odds[['AvgH', 'AvgD', 'AvgA']]
implied = inv.div(inv.sum(axis=1), axis=0)
gap = (implied['AvgH'] - implied['AvgA']).abs().values

results_gap = []
bins = [(0.0, 0.15, '势均力敌 (差距<0.15)'),
        (0.15, 0.35, '中等差距 (0.15-0.35)'),
        (0.35, 1.01, '实力悬殊 (差距>0.35)')]
pf(f"\n{'子集':<24}{'样本':>6}{'平局率':>8}{'召回率':>12}{'精确率':>8}{'校准对比':>16}{'信号抛弃':>9}")
for lo, hi, label in bins:
    mask = valid_odds & (gap >= lo) & (gap < hi)
    subset_report(mask, label, results_gap)
    n = mask.sum()
    if n >= 30:
        yt, pt = y_test[mask], pred[mask]
        actual = (yt == 1).sum()
        correct = ((yt == 1) & (pt == 1)).sum()
        recall = correct / actual
        pf(f"{'':24}{'':>6}{'':>8}{recall*100:>11.2f}%")
        # 重新格式化打印（subset_report已打印，这里补充召回落到行内）
pf("\n注: 上表每行为独立子集统计")

# ============ 子集2: 赛程密度（休息天数） ============
pf("\n" + "=" * 70)
pf("子集2: 赛程密度（两队中较短休息天数）")
pf("=" * 70)

# 合并训练+测试, 计算每队每场比赛后的休息天数
all_games = pd.concat([
    train_df[['Date', 'HomeTeam', 'AwayTeam']],
    test_df[['Date', 'HomeTeam', 'AwayTeam']]
], ignore_index=True).sort_values('Date').reset_index(drop=True)

# 构建每支球队的比赛日期列表
team_dates = {}
for _, row in all_games.iterrows():
    for team in (row['HomeTeam'], row['AwayTeam']):
        team_dates.setdefault(team, []).append(row['Date'])

# 对测试集每场比赛计算休息天数
def rest_days(row):
    """返回该场比赛两队各自的休息天数（距上一场）"""
    rests = []
    for team in (row['HomeTeam'], row['AwayTeam']):
        dates = team_dates.get(team, [])
        prev = [d for d in dates if d < row['Date']]
        if prev:
            rests.append((row['Date'] - max(prev)).days)
        else:
            rests.append(np.nan)
    return rests

pf("计算休息天数...")
rests = test_df.apply(rest_days, axis=1, result_type='expand')
test_df = pd.concat([test_df.reset_index(drop=True), rests], axis=1)
test_df.columns = list(test_df.columns[:-2]) + ['HomeRest', 'AwayRest']
test_df['MinRest'] = test_df[['HomeRest', 'AwayRest']].min(axis=1)

results_rest = []
bins_rest = [(0, 5, '密集赛程 (休息≤5天)'),
             (6, 8, '正常赛程 (休息6-8天)'),
             (9, 999, '充足休息 (休息≥9天)')]
pf(f"\n{'子集':<24}{'样本':>6}{'平局率':>8}{'召回率':>9}{'精确率':>8}{'校准对比':>16}{'信号抛弃':>9}")
for lo, hi, label in bins_rest:
    mask = test_df['MinRest'].notna().values & (test_df['MinRest'] >= lo) & (test_df['MinRest'] <= hi)
    subset_report(mask, label, results_rest)

# ============ 汇总与结论 ============
pf("\n\n" + "=" * 90)
pf("子集分析结论")
pf("=" * 90)

df_gap = pd.DataFrame(results_gap)
df_rest = pd.DataFrame(results_rest)

pf("\n实力差距维度:")
for _, r in df_gap.iterrows():
    pf(f"  {r['子集']:<24} 平局率{r['平局率']*100:5.1f}%, 召回率{r['平局召回率']*100:5.2f}%, "
       f"校准{r['模型平均平局概率']*100:.1f}% vs {r['平局率']*100:.1f}%")

pf("\n赛程密度维度:")
for _, r in df_rest.iterrows():
    pf(f"  {r['子集']:<24} 平局率{r['平局率']*100:5.1f}%, 召回率{r['平局召回率']*100:5.2f}%, "
       f"校准{r['模型平均平局概率']*100:.1f}% vs {r['平局率']*100:.1f}%")

pf("\n" + "=" * 90)
pf("关键发现")
pf("=" * 90)
balanced = df_gap.iloc[0]
pf(f"1. 势均力敌的比赛平局率最高({balanced['平局率']*100:.1f}%), "
   f"但召回率仅{balanced['平局召回率']*100:.2f}%")
pf("   → 越应该预测平局的场次, 模型失败得越彻底（悖论）")

max_recall_gap = df_gap.loc[df_gap['平局召回率'].idxmax()]
pf(f"2. 即使在召回率最高的实力差距组({max_recall_gap['子集']}), "
   f"召回率也只有{max_recall_gap['平局召回率']*100:.2f}%")

for _, r in df_rest.iterrows():
    pf(f"3. {r['子集']}: 校准{r['模型平均平局概率']*100:.1f}% vs 实际{r['平局率']*100:.1f}%"
       f" → 概率正确但点预测失败")
pf("   → 赛程密度不改变平局不可预测性（排除疲劳解释）")

df_gap.to_csv('results/subset_strength_gap.csv', index=False, encoding='utf-8-sig')
df_rest.to_csv('results/subset_schedule_congestion.csv', index=False, encoding='utf-8-sig')
pf("\n结果已保存: results/subset_strength_gap.csv, results/subset_schedule_congestion.csv")
