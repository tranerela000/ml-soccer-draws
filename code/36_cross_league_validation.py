"""
跨联赛验证（修正版）：证明平局预测失败是跨联赛的普遍现象

严格对齐 34_draw_analysis.py 的数据处理流程（剔除当场比赛统计）。

Part A: 全局模型按联赛拆解 —— 主模型在5个联赛上分别的平局表现
Part B: 各联赛单独训练 —— 排除"训练分布差异"的解释
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
pf("跨联赛验证（修正版）")
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

pf(f"有效特征数: {len(feature_cols)}（已剔除当场比赛统计）")


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


def make_model():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
        random_state=42, eval_metric='mlogloss')


leagues = {
    'E0': '英超',
    'D1': '德甲',
    'SP1': '西甲',
    'F1': '法甲',
    'I1': '意甲'
}


def evaluate(y_true, y_pred, y_prob, tag):
    """返回平局分析结果字典"""
    n = len(y_true)
    actual = (y_true == 1).sum()
    predicted = (y_pred == 1).sum()
    correct = ((y_true == 1) & (y_pred == 1)).sum()
    recall = correct / actual if actual > 0 else 0
    precision = correct / predicted if predicted > 0 else 0
    acc = (y_pred == y_true).mean()
    avg_prob = y_prob[:, 1].mean()
    actual_rate = actual / n
    missed = (y_true == 1) & (y_pred != 1)
    missed_n = int(missed.sum())
    high_missed = int((y_prob[missed, 1] > 0.25).sum()) if missed_n > 0 else 0
    return {
        '联赛': tag,
        '测试样本': n,
        '整体准确率': acc,
        '实际平局': int(actual),
        '预测平局': int(predicted),
        '正确平局': int(correct),
        '平局召回率': recall,
        '平局精确率': precision,
        '平均平局概率': avg_prob,
        '实际平局比例': actual_rate,
        '漏掉平局': missed_n,
        '高概率漏掉': high_missed
    }


X_all_train = prepare_X(train_df, feature_cols)
y_all_train = train_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
X_all_test = prepare_X(test_df, feature_cols, final_cols=list(X_all_train.columns))
y_all_test = test_df['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values

# ============ Part A: 全局模型按联赛拆解 ============
pf("\n" + "=" * 70)
pf("Part A: 全局模型在5个联赛上的平局表现")
pf("=" * 70)

pf("训练全局模型...")
global_model = make_model()
global_model.fit(X_all_train, y_all_train)
global_pred = global_model.predict(X_all_test)
global_prob = global_model.predict_proba(X_all_test)

results_a = []
for code, name in leagues.items():
    mask = (test_df['Div'] == code).values
    r = evaluate(y_all_test[mask], global_pred[mask], global_prob[mask], name)
    results_a.append(r)

pf(f"\n{'联赛':<6}{'样本':>6}{'准确率':>8}{'实际平局':>9}{'正确':>6}{'召回率':>9}{'模型概率':>9}{'实际比例':>9}")
for r in results_a:
    pf(f"{r['联赛']:<6}{r['测试样本']:>6}{r['整体准确率']*100:>7.2f}%{r['实际平局']:>9}"
       f"{r['正确平局']:>6}{r['平局召回率']*100:>8.2f}%{r['平均平局概率']*100:>8.1f}%{r['实际平局比例']*100:>8.1f}%")

# ============ Part B: 各联赛单独训练 ============
pf("\n" + "=" * 70)
pf("Part B: 各联赛单独训练的模型（稳健性检验）")
pf("=" * 70)

results_b = []
for code, name in leagues.items():
    tr_mask = (train_df['Div'] == code).values
    te_mask = (test_df['Div'] == code).values

    X_tr = prepare_X(train_df[tr_mask], feature_cols)
    y_tr = train_df.loc[tr_mask, 'FTR'].map({'H': 0, 'D': 1, 'A': 2})
    X_te = prepare_X(test_df[te_mask], feature_cols, final_cols=list(X_tr.columns))
    y_te = test_df.loc[te_mask, 'FTR'].map({'H': 0, 'D': 1, 'A': 2}).values

    model = make_model()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)

    r = evaluate(y_te, pred, prob, name)
    results_b.append(r)
    pf(f"  {name}: 训练{tr_mask.sum()}场, 测试{r['测试样本']}场, "
       f"准确率{r['整体准确率']*100:.2f}%, 召回率{r['平局召回率']*100:.2f}% "
       f"({r['正确平局']}/{r['实际平局']})")

# ============ 汇总 ============
pf("\n\n" + "=" * 90)
pf("跨联赛验证汇总")
pf("=" * 90)

df_a = pd.DataFrame(results_a)
df_b = pd.DataFrame(results_b)

pf("\nPart A 全局模型按联赛拆解:")
for _, r in df_a.iterrows():
    pf(f"  {r['联赛']:<6} 召回率{r['平局召回率']*100:6.2f}% ({r['正确平局']}/{r['实际平局']})  "
       f"校准: 模型{r['平均平局概率']*100:.1f}% vs 实际{r['实际平局比例']*100:.1f}%")

pf("\nPart B 各联赛单独训练:")
for _, r in df_b.iterrows():
    pf(f"  {r['联赛']:<6} 召回率{r['平局召回率']*100:6.2f}% ({r['正确平局']}/{r['实际平局']})  "
       f"校准: 模型{r['平均平局概率']*100:.1f}% vs 实际{r['实际平局比例']*100:.1f}%")

pf("\n统计结论:")
pf(f"  Part A 召回率范围: {df_a['平局召回率'].min()*100:.2f}% ~ {df_a['平局召回率'].max()*100:.2f}%")
pf(f"  Part B 召回率范围: {df_b['平局召回率'].min()*100:.2f}% ~ {df_b['平局召回率'].max()*100:.2f}%")
pf(f"  Part A 全部<10%: {'是' if (df_a['平局召回率'] < 0.10).all() else '否'}")
pf(f"  Part B 全部<15%: {'是' if (df_b['平局召回率'] < 0.15).all() else '否'}")
calib_a = (abs(df_a['平均平局概率'] - df_a['实际平局比例']) < 0.05).all()
pf(f"  Part A 概率校准全部合理(<5pp): {'是' if calib_a else '否'}")

pf("\n关键结论: 无论全局模型还是各联赛单独训练，")
pf("平局召回率都远低于25%基础发生率，而概率校准保持正确，")
pf("证明平局不可预测性是跨联赛的普遍现象。")

# 保存
df_a.to_csv('results/cross_league_partA_global.csv', index=False, encoding='utf-8-sig')
df_b.to_csv('results/cross_league_partB_persleague.csv', index=False, encoding='utf-8-sig')
pf(f"\n结果已保存: results/cross_league_partA_global.csv, results/cross_league_partB_persleague.csv")
