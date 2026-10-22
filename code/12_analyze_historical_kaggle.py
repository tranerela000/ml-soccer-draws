"""
分析Kaggle数据集中2020-2025年的历史数据覆盖情况
"""
import pandas as pd
import os

# 数据路径
data_dir = 'data'
matches_file = os.path.join(data_dir, 'Matches.csv')

# 加载数据
print("正在加载Kaggle Matches.csv...")
df_matches = pd.read_csv(matches_file, low_memory=False)

# 清理BOM字符
df_matches.columns = df_matches.columns.str.replace('ï»¿', '')

# 转换日期列
print("转换日期格式...")
df_matches['MatchDate'] = pd.to_datetime(df_matches['MatchDate'])

# 筛选2020-2025年五大联赛数据（赛季跨年：如2020-21赛季从2020年8月到2021年5月）
print("\n筛选2020-2025年五大联赛数据...")
df_historical = df_matches[
    (df_matches['MatchDate'] >= '2020-08-01') & 
    (df_matches['MatchDate'] < '2025-08-01') &
    (df_matches['Division'].isin(['E0', 'SP1', 'I1', 'D1', 'F1']))
].copy()

print(f"\n✅ 2020-2025年五大联赛总比赛数: {len(df_historical):,}")

# 按联赛统计
print("\n按联赛分布:")
league_counts = df_historical['Division'].value_counts()
league_names = {
    'E0': '英超',
    'SP1': '西甲',
    'I1': '意甲',
    'D1': '德甲',
    'F1': '法甲'
}
for league, count in league_counts.items():
    print(f"  {league_names.get(league, league)}: {count:,}")

# 添加赛季标签
df_historical['Season'] = df_historical['MatchDate'].apply(
    lambda x: f"{x.year}-{(x.year+1) % 100:02d}" if x.month >= 8 else f"{x.year-1}-{x.year % 100:02d}"
)

# 按赛季统计
print("\n按赛季分布:")
season_counts = df_historical['Season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"  {season}: {count:,}")

# 检查关键特征完整性
print("\n关键特征缺失率:")
key_features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form3Away', 'Form5Home', 'Form5Away']
for feature in key_features:
    missing_pct = df_historical[feature].isna().mean() * 100
    print(f"  {feature}: {missing_pct:.2f}%")

# 保存历史数据
output_file = os.path.join(data_dir, 'kaggle_2020_2025.csv')
df_historical.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 已保存到: {output_file}")

# 显示数据示例
print("\n数据示例:")
print(df_historical[['Season', 'MatchDate', 'Division', 'HomeTeam', 'AwayTeam', 'FTHome', 'FTAway', 'HomeElo', 'AwayElo']].head(10))
