"""
论文图表生成（Figure 1-4）
复用 34_draw_analysis.py 的数据处理流程，重训参考模型获取逐场预测，
再结合 results/*.csv 生成投稿图表。

Fig 1: 混淆矩阵热图（参考模型, 2025-26测试集）
Fig 2: 校准 vs 召回率散点图（算法×联赛×赛季）
Fig 3: 势均力敌子集 P(draw) 分布直方图（悖论可视化）
Fig 4: 时间趋势（召回率柱状 + 信号抛弃比例折线）
"""
import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def pf(msg):
    print(msg)
    sys.stdout.flush()


plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10.5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 100,
})

# ============ 数据加载（与34完全一致） ============
pf("=" * 70)
pf("论文图表生成")
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
n_draws = int((y_test == 1).sum())
pf(f"训练集: {len(X_train)}, 测试集: {n_test}, 平局: {n_draws}")

# ============ 重训参考模型 ============
model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
    random_state=42, eval_metric='mlogloss')
model.fit(X_train, y_train)
pred = model.predict(X_test)
probs = model.predict_proba(X_test)

acc = (pred == y_test).mean()
draw_pred = int((pred == 1).sum())
draw_tp = int(((pred == 1) & (y_test == 1)).sum())
avg_draw_prob = probs[:, 1].mean()

pf(f"验证: 准确率{acc*100:.2f}% (期望53.48), "
   f"召回率{draw_tp/n_draws*100:.2f}% (期望1.79), "
   f"预测平局{draw_pred}场 (期望33), "
   f"平均P(draw){avg_draw_prob*100:.1f}% (期望26.2)")

# ============ Fig 1: 混淆矩阵 ============
cm = confusion_matrix(y_test, pred)
labels_en = ['Home win', 'Draw', 'Away win']

fig, ax = plt.subplots(figsize=(6.2, 5.2))
# 行归一化着色，格内显示 计数 + 行百分比
row_sum = cm.sum(axis=1, keepdims=True)
norm = cm / row_sum
im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        color = 'white' if norm[i, j] > 0.5 else 'black'
        weight = 'bold' if i == j else 'normal'
        # 平局行/列用颜色强调"吸收"现象
        ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]*100:.1f}%)",
                ha='center', va='center', color=color, fontsize=13, weight=weight)

# 高亮平局行（漏掉98.2%）与平局列（仅33次预测）
ax.add_patch(plt.Rectangle((-0.5, 0.5), 3, 1, fill=False,
                           edgecolor='#d62728', lw=2.5, ls='--'))
ax.add_patch(plt.Rectangle((0.5, -0.5), 1, 3, fill=False,
                           edgecolor='#2ca02c', lw=2.5, ls=':'))

ax.set_xticks(range(3), ['Pred. home win', 'Pred. draw', 'Pred. away win'])
ax.set_yticks(range(3), ['Actual home win', 'Actual draw', 'Actual away win'])
ax.set_title('Confusion matrix — reference model (2025-26 test set)')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row proportion')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig1_confusion_matrix.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig1_confusion_matrix.pdf', bbox_inches='tight')
plt.close(fig)
pf("Fig 1 混淆矩阵 已保存")

# ============ 势均力敌子集（Fig 3 数据准备） ============
inv = 1.0 / test_df[['AvgH', 'AvgD', 'AvgA']]
implied = inv.div(inv.sum(axis=1), axis=0)
gap = (implied['AvgH'] - implied['AvgA']).abs().values
balanced = (gap < 0.15) & test_df[['AvgH', 'AvgD', 'AvgA']].notna().all(axis=1).values

bal_idx = np.where(balanced)[0]
bal_y = y_test.values[bal_idx]
bal_pdraw = probs[bal_idx, 1]
bal_draws = (bal_y == 1).sum()
bal_pred_draw = (bal_pdraw.values if hasattr(bal_pdraw, 'values') else bal_pdraw)
caught = (bal_y == 1) & (pred[bal_idx] == 1)
n_caught = int(caught.sum())
missed_probs = bal_pdraw[(bal_y == 1) & (pred[bal_idx] != 1)]
supp = (missed_probs > 0.25).mean()

pf(f"势均力敌子集: {len(bal_idx)}场, 平局{bal_draws}场 "
   f"({bal_draws/len(bal_idx)*100:.1f}%), 抓到{n_caught}场, "
   f"信号抛弃{supp*100:.1f}%")

# ============ Fig 3: 势均力敌子集 P(draw) 分布 ============
fig, ax = plt.subplots(figsize=(7.2, 4.8))
bins = np.linspace(0, 0.55, 29)
ax.hist(bal_pdraw[bal_y != 1], bins=bins, alpha=0.55, color='#7f7f7f',
        label=f'Non-draws in subset (n={int((bal_y != 1).sum())})', edgecolor='white')
ax.hist(bal_pdraw[bal_y == 1], bins=bins, alpha=0.75, color='#d62728',
        label=f'Actual draws in subset (n={bal_draws})', edgecolor='white')

ax.axvline(0.25, color='#1f77b4', ls='--', lw=2,
           label='Base-rate threshold (0.25)')
# 标注被抓到的6场
ax.scatter(bal_pdraw[caught], np.full(n_caught, 3.5), marker='v', s=90,
           color='#2ca02c', zorder=5, label=f'Draws caught by argmax (n={n_caught})')

# 高亮 >0.25 区域
ax.axvspan(0.25, 0.55, color='#d62728', alpha=0.06)
ax.annotate(f'{supp*100:.0f}% of missed draws\nlie above 0.25',
            xy=(0.36, ax.get_ylim()[1]*0.62), fontsize=12, color='#d62728',
            weight='bold', ha='center')

ax.set_xlabel('Model P(draw) in balanced matches (prob. gap < 0.15)')
ax.set_ylabel('Number of matches')
ax.set_title('Balanced matches: the draw signal exists, but argmax never acts')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig3_balanced_subset_distribution.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig3_balanced_subset_distribution.pdf', bbox_inches='tight')
plt.close(fig)
pf("Fig 3 子集概率分布 已保存")

# ============ Fig 2: 校准 vs 召回率散点 ============
alg = pd.read_csv('results/multi_algorithm_validation.csv')
alg = alg[alg['算法'] != 'XGBoost (reference)']  # 参考模型单独标注
leag_a = pd.read_csv('results/cross_league_partA_global.csv')
leag_b = pd.read_csv('results/cross_league_partB_persleague.csv')
trend = pd.read_csv('results/time_trend_analysis.csv')

fig, ax = plt.subplots(figsize=(9.0, 5.4))

sc_alg = ax.scatter(alg['平均平局概率']*100, alg['平局召回率']*100,
                    s=130, marker='o', color='#1f77b4', edgecolor='black',
                    lw=0.8, label='Algorithms (n=4 + reference below)', zorder=4)
sc_sea = ax.scatter(trend['平均平局概率']*100, trend['平局召回率']*100,
                    s=110, marker='s', color='#2ca02c', edgecolor='black',
                    lw=0.8, label='Seasons (rolling origin)', zorder=4)
sc_la = ax.scatter(leag_a['平均平局概率']*100, leag_a['平局召回率']*100,
                   s=110, marker='^', facecolor='#ff7f0e', edgecolor='black',
                   lw=0.8, label='Leagues (global model)', zorder=4)
sc_lb = ax.scatter(leag_b['平均平局概率']*100, leag_b['平局召回率']*100,
                   s=110, marker='v', facecolor='none', edgecolor='#9467bd',
                   lw=1.6, label='Leagues (league-specific models)', zorder=4)

# 参考模型 + 势均力敌子集 高亮
ax.scatter([avg_draw_prob*100], [draw_tp/n_draws*100], s=210, marker='*',
           color='#d62728', edgecolor='black', lw=0.8,
           label='Reference model (XGBoost)', zorder=6)
ax.scatter([np.mean(bal_pdraw[bal_y == 1])*100], [n_caught/bal_draws*100],
           s=150, marker='D', color='#e377c2', edgecolor='black', lw=0.8,
           label='Balanced subset (the paradox)', zorder=6)

ax.axhline(25.5, color='gray', ls='--', lw=1.5)
ax.axvline(25.5, color='gray', ls='--', lw=1.5)
ax.text(20.4, 26.2, 'random-proportional recall', fontsize=9.5, color='gray')
ax.text(25.75, 29.3, 'base rate', fontsize=9.5, color='gray', rotation=90, va='top')

ax.set_xlabel('Mean predicted P(draw) (%)  — calibration')
ax.set_ylabel('Draw recall (%)')
ax.set_title('Calibration vs. classification: every point is well calibrated,\nalmost none catches draws')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
ax.set_xlim(20, 32)
ax.set_ylim(-1, 30)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig2_calibration_vs_recall.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig2_calibration_vs_recall.pdf', bbox_inches='tight')
plt.close(fig)
pf("Fig 2 校准vs召回率 已保存")

# ============ Fig 4: 时间趋势 ============
fig, ax = plt.subplots(figsize=(7.4, 4.8))
seasons = trend['赛季'].tolist()
x = np.arange(len(seasons))

bars = ax.bar(x, trend['平局召回率']*100, width=0.55, color='#1f77b4',
              alpha=0.85, label='Draw recall', edgecolor='black', lw=0.6)
# 基础发生率（各赛季）
ax.plot(x, trend['实际平局比例']*100, 'o--', color='#d62728', lw=2, ms=8,
        label='Observed draw rate (base rate)')
for xi, (rec, base) in enumerate(zip(trend['平局召回率']*100, trend['实际平局比例']*100)):
    ax.text(xi, rec + 0.7, f'{rec:.1f}%', ha='center', fontsize=10.5, weight='bold')

ax2 = ax.twinx()
ax2.plot(x, trend['高概率漏掉']/trend['漏掉平局']*100, 's-', color='#ff7f0e',
         lw=2, ms=7, label='Signal suppression ratio (right)')
ax2.set_ylabel('Suppression ratio (%)', color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')
ax2.set_ylim(0, 100)

ax.set_xticks(x, seasons)
ax.set_ylabel('Rate (%)')
ax.set_title('Draw recall stays below half the base rate in every season\n(test season; model trained on all prior seasons)')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.08),
          ncol=3, frameon=False, fontsize=10.5)
ax.set_ylim(0, 30)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig4_time_trend.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/fig4_time_trend.pdf', bbox_inches='tight')
plt.close(fig)
pf("Fig 4 时间趋势 已保存")

# ============ 数字核验汇总 ============
pf("\n" + "=" * 70)
pf("核验汇总（与论文正文对照）")
pf("=" * 70)
pf(f"准确率: {acc*100:.2f}%  | 论文: 53.48%")
pf(f"召回率: {draw_tp}/{n_draws} = {draw_tp/n_draws*100:.2f}%  | 论文: 8/446 = 1.79%")
pf(f"预测平局: {draw_pred}场  | 论文: 33场")
pf(f"平均P(draw): {avg_draw_prob*100:.1f}%  | 论文: 26.2%")
pf(f"势均力敌: {len(bal_idx)}场 / 平局率{bal_draws/len(bal_idx)*100:.1f}% (论文29.8) / "
   f"召回{n_caught}/{bal_draws} (论文6/164) / 信号抛弃{supp*100:.1f}% (论文95.1)")
pf(f"\n全部图表已保存至 {FIG_DIR}/ (PNG 300dpi + PDF矢量)")
