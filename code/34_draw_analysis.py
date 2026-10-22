"""
平局预测详细诊断
回答两个问题:
1. 预测为平局时,多少次真的平局?(精确率)
2. 真的平局时,多少次被抓到?(召回率)
"""
import pandas as pd
import numpy as np
import sys
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

def pf(msg):
    print(msg)
    sys.stdout.flush()

pf("=" * 70)
pf("平局预测详细诊断（2025-26赛季, 1752场）")
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

X_all = prepare_X(train_df, feature_cols)
y_all = train_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
X_test = prepare_X(test_df, feature_cols, final_cols=list(X_all.columns))
y_test = test_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

model = XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=20,
    random_state=42, eval_metric='mlogloss')
model.fit(X_all, y_all)
pred = model.predict(X_test)
probs = model.predict_proba(X_test)

labels = ['主胜H', '平局D', '客胜A']
cm = confusion_matrix(y_test, pred)

pf("\n实际分布: 主胜{}场({:.1f}%), 平局{}场({:.1f}%), 客胜{}场({:.1f}%)".format(
    *(lambda vc: [vc.get(i, 0) for i in range(3)] + [vc.get(i, 0)/len(y_test)*100 for i in range(3)])(pd.Series(y_test).value_counts())))

pf("\n混淆矩阵（行=实际, 列=预测）:")
pf(f"{'':>10}{'预测主胜':>10}{'预测平局':>10}{'预测客胜':>10}")
for i, name in enumerate(labels):
    pf(f"{'实际'+name:>10}{cm[i][0]:>10}{cm[i][1]:>10}{cm[i][2]:>10}")

# 平局两类错误
tp = cm[1][1]
fp = cm[0][1] + cm[2][1]      # 预测平局但实际不是
fn = cm[1][0] + cm[1][2]      # 实际平局但没预测出来
prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)

pf("\n" + "=" * 70)
pf("平局的两种错误")
pf("=" * 70)
pf(f"问题1: 预测平局时,真的是平局吗?")
pf(f"  模型预测平局 {tp+fp} 次, 其中真的平局 {tp} 次 → 精确率 {prec*100:.1f}%")
pf(f"  (即预测平局的比赛中, {(1-prec)*100:.1f}% 其实分出了胜负)")
pf(f"\n问题2: 真的平局时,模型抓到了吗?")
pf(f"  实际平局共 {tp+fn} 场, 模型只抓到 {tp} 场 → 召回率 {rec*100:.1f}%")
pf(f"  (即 {(fn/(tp+fn))*100:.1f}% 的平局被模型误判为主胜或客胜)")

# 模型给平局的概率分布
pf("\n模型平均给平局的概率: {:.1f}% (实际平局率 {:.1f}%)".format(
    probs[:, 1].mean() * 100, (y_test == 1).mean() * 100))
pf("平局概率是三者最高时才预测平局, 这种情况只发生 {:.1f}% 的比赛".format(
    (pred == 1).mean() * 100))

# 赔率基准同样情况
odds_cols = ['AvgD']
valid = test_df[['AvgH', 'AvgD', 'AvgA']].notna().all(axis=1).values
odds_pred = np.argmin(test_df[['AvgH', 'AvgD', 'AvgA']].values, axis=1)
odds_draw_pred = (odds_pred == 1).sum()
odds_tp = ((odds_pred == 1) & (y_test[valid] == 1)).sum()
pf("\n对照: 赔率把平局当热门的比赛只有 {} 场 ({:.1f}%), 其中真平局 {} 场".format(
    odds_draw_pred, odds_draw_pred / len(test_df) * 100, odds_tp))

# 提高平局召回的代价演示
pf("\n" + "=" * 70)
pf("如果强行多预测平局会怎样?")
pf("=" * 70)
for thresh in [0.26, 0.30, 0.33]:
    forced = probs.copy()
    pred_f = np.where(forced[:, 1] >= thresh, 1, forced.argmax(axis=1))
    tp_f = ((pred_f == 1) & (y_test == 1)).sum()
    n_draw_pred = (pred_f == 1).sum()
    acc_f = (pred_f == y_test).mean()
    p_f = tp_f / max(n_draw_pred, 1)
    pf(f"  平局概率>={thresh:.0%}就预测平局: 预测平局{n_draw_pred}次, "
       f"精确率{p_f*100:.1f}%, 召回率{tp_f/(y_test==1).sum()*100:.1f}%, 总准确率降至{acc_f*100:.2f}%")
