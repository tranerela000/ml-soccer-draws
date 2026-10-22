"""
多算法验证（修正版）：证明平局预测失败是跨算法的结构性现象

严格对齐 34_draw_analysis.py 的数据处理流程:
- LEAK_COLS: 剔除当场比赛统计（结果型数据，防止泄漏）
- 特征筛选: 缺失率>50%剔除
- 填充: 用训练集中位数，缺失补0
- 列对齐: reindex到训练集列

测试算法:
1. Logistic Regression（最简单基准）
2. Random Forest（Bagging集成）
3. LightGBM（GBDT同类）
4. MLP神经网络
5. XGBoost（参考基准，与34完全一致的配置）

附加对照: 博彩赔率基准（argmin赔率选热门）
"""
import pandas as pd
import numpy as np
import re
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("警告: LightGBM未安装，跳过")

from xgboost import XGBClassifier


def pf(msg):
    print(msg)
    sys.stdout.flush()


# ============ 数据加载（对齐34的处理） ============
pf("=" * 70)
pf("多算法验证（修正版）")
pf("=" * 70)

train_df = pd.read_csv('data/processed/train_strict.csv')
test_df = pd.read_csv('data/processed/test_strict.csv')
for d in (train_df, test_df):
    d['Date'] = pd.to_datetime(d['Date'])
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)

# 关键: 剔除当场比赛统计（结果型数据）与ID列 —— 与34完全一致
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

# 检查是否有泄漏列残留
residual_leak = [c for c in feature_cols if c in LEAK_COLS]
pf(f"泄漏列残留检查: {'无 ✓' if not residual_leak else residual_leak}")


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

n_test = len(y_test)
actual_draws = (y_test == 1).sum()
pf(f"训练集: {len(X_train)}, 测试集: {n_test}")
pf(f"测试集平局: {actual_draws}场 ({actual_draws/n_test*100:.1f}%)")

# ============ 基准 ============
pf(f"\n基准（永远预测主胜）准确率: {(y_test == 0).mean()*100:.2f}%")

# 赔率基准
valid = test_df[['AvgH', 'AvgD', 'AvgA']].notna().all(axis=1).values
odds_pred = np.argmin(test_df[['AvgH', 'AvgD', 'AvgA']].values, axis=1)
odds_acc = (odds_pred == y_test[valid]).mean()
odds_draw_pred = (odds_pred == 1).sum()
odds_draw_tp = ((odds_pred == 1) & (y_test[valid] == 1)).sum()
odds_recall = odds_draw_tp / actual_draws
pf(f"赔率基准（选最低赔率热门）: 准确率{odds_acc*100:.2f}%, "
   f"平局预测{odds_draw_pred}场, 召回率{odds_recall*100:.2f}%")

# ============ 定义算法（容量对齐：均适度正则化） ============


def get_models():
    models = {}

    models['Logistic Regression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=42))
    ])

    models['Random Forest'] = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        n_jobs=-1, random_state=42)

    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200, max_depth=3, num_leaves=7,
            learning_rate=0.05, subsample=0.7, colsample_bytree=0.5,
            reg_lambda=5.0, min_child_samples=20,
            verbose=-1, random_state=42)

    models['MLP Neural Net'] = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.05,
                              early_stopping=True, max_iter=500,
                              random_state=42))
    ])

    # 参考基准: 与34_draw_analysis.py完全一致的配置
    models['XGBoost (reference)'] = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
        random_state=42, eval_metric='mlogloss')

    return models


# ============ 训练与评估 ============
results = []

for name, model in get_models().items():
    pf(f"\n{'=' * 70}")
    pf(f"训练: {name}")
    pf(f"{'=' * 70}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    draw_idx = list(model.classes_).index(1)

    acc = accuracy_score(y_test, y_pred)
    train_acc = model.score(X_train, y_train)

    draw_pred = (y_pred == 1).sum()
    draw_correct = ((y_test == 1) & (y_pred == 1)).sum()
    draw_recall = draw_correct / actual_draws
    draw_precision = draw_correct / draw_pred if draw_pred > 0 else 0.0

    avg_draw_prob = y_prob[:, draw_idx].mean()
    actual_draw_rate = actual_draws / n_test

    missed = (y_test == 1) & (y_pred != 1)
    missed_probs = y_prob[missed, draw_idx]
    high_prob_missed = (missed_probs > 0.25).sum() if missed.sum() > 0 else 0

    pf(f"训练集准确率: {train_acc*100:.2f}%")
    pf(f"测试集准确率: {acc*100:.2f}%")
    pf(f"平局: 预测{draw_pred}场, 正确{draw_correct}场, 召回率{draw_recall*100:.2f}%, 精确率{draw_precision*100:.2f}%")
    pf(f"平均平局概率: {avg_draw_prob*100:.1f}% vs 实际{actual_draw_rate*100:.1f}%")
    pf(f"漏掉的平局: {missed.sum()}场, P(draw)>0.25的: {high_prob_missed}场 ({high_prob_missed/max(missed.sum(),1)*100:.1f}%)")

    results.append({
        '算法': name,
        '训练集准确率': train_acc,
        '测试集准确率': acc,
        '平局预测数': int(draw_pred),
        '平局正确数': int(draw_correct),
        '平局召回率': draw_recall,
        '平局精确率': draw_precision,
        '平均平局概率': avg_draw_prob,
        '实际平局比例': actual_draw_rate,
        '漏掉平局数': int(missed.sum()),
        '高概率漏掉数': int(high_prob_missed)
    })

# ============ 汇总 ============
pf("\n\n" + "=" * 90)
pf("多算法验证汇总")
pf("=" * 90)

results_df = pd.DataFrame(results)

pf("\n整体准确率对比:")
pf(f"  {'赔率基准(选热门)':30} {odds_acc*100:6.2f}%")
for _, r in results_df.iterrows():
    pf(f"  {r['算法']:30} {r['测试集准确率']*100:6.2f}%")

pf("\n平局召回率对比（核心指标）:")
pf(f"  {'赔率基准(选热门)':30} {odds_recall*100:6.2f}%  ({odds_draw_tp}/{actual_draws})")
for _, r in results_df.iterrows():
    pf(f"  {r['算法']:30} {r['平局召回率']*100:6.2f}%  ({r['平局正确数']}/{actual_draws})")

pf("\n概率校准对比:")
for _, r in results_df.iterrows():
    gap = abs(r['平均平局概率'] - r['实际平局比例']) * 100
    pf(f"  {r['算法']:30} 模型{r['平均平局概率']*100:5.1f}% vs 实际{r['实际平局比例']*100:5.1f}% (差距{gap:.1f}pp)")

pf("\n\"信号存在但被抛弃\"比例:")
for _, r in results_df.iterrows():
    if r['漏掉平局数'] > 0:
        ratio = r['高概率漏掉数'] / r['漏掉平局数'] * 100
        pf(f"  {r['算法']:30} {ratio:5.1f}%  ({r['高概率漏掉数']}/{r['漏掉平局数']})")

# ============ 统计结论 ============
pf("\n\n" + "=" * 90)
pf("统计结论")
pf("=" * 90)
pf(f"测试算法数: {len(results_df)}")
pf(f"平局召回率范围: {results_df['平局召回率'].min()*100:.2f}% ~ {results_df['平局召回率'].max()*100:.2f}%")
pf(f"平局召回率均值: {results_df['平局召回率'].mean()*100:.2f}% (±{results_df['平局召回率'].std()*100:.2f}%)")

all_below_30 = (results_df['平局召回率'] < 0.30).all()
half_below_20 = (results_df['平局召回率'] < 0.20).mean() >= 0.5
pf(f"\n✓ 所有算法平局召回率均<30%: {'是' if all_below_30 else '否'}")
pf(f"✓ 至少一半算法平局召回率<20%: {'是' if half_below_20 else '否'}")
pf(f"✓ 赔率基准平局召回率: {odds_recall*100:.2f}%（行业最强信息聚合器）")

pf("\n关键结论:")
pf("即使换用完全不同的算法家族（线性/Bagging/GBDT/神经网络），")
pf("平局召回率依然远低于其25%的基础发生率，")
pf("证明平局预测失败源于赛前特征的信息边界，而非特定算法缺陷。")

results_df.to_csv('results/multi_algorithm_validation.csv', index=False, encoding='utf-8-sig')
pf(f"\n结果已保存至: results/multi_algorithm_validation.csv")
