"""
合并2020-2025年历史数据：Kaggle (ELO+Form) + Football-Data.co.uk (赔率+统计)
"""
import pandas as pd
import numpy as np
import os

# 数据路径
data_dir = 'data'

# 加载数据
print("正在加载数据...")
df_kaggle = pd.read_csv(os.path.join(data_dir, 'kaggle_2020_2025.csv'))
df_fd = pd.read_csv(os.path.join(data_dir, 'footballdata_uk_2020_2025.csv'), low_memory=False)

# 清理BOM字符
df_kaggle.columns = df_kaggle.columns.str.replace('ï»¿', '')
df_fd.columns = df_fd.columns.str.replace('ï»¿', '')

print(f"Kaggle数据: {len(df_kaggle):,} 场")
print(f"Football-Data数据: {len(df_fd):,} 场")

# 球队名称标准化（使用之前的映射表）
def normalize_team_names():
    return {
        'Man United': 'Manchester Utd', 'Man City': 'Manchester City',
        'Ath Madrid': 'Atletico Madrid', 'Ath Bilbao': 'Ath Bilbao',
        'Nottingham': "Nott'm Forest", 'Brighton': 'Brighton',
        'Wolves': 'Wolves', 'Newcastle': 'Newcastle',
        'Tottenham': 'Tottenham', 'West Ham': 'West Ham',
        'Leicester': 'Leicester', 'Leeds': 'Leeds United',
        'Dortmund': 'Dortmund', 'Leverkusen': 'Leverkusen',
        'M\'gladbach': "M'gladbach", 'Stuttgart': 'Stuttgart',
        'Inter': 'Inter', 'AS Roma': 'Roma',
        'Verona': 'Verona', 'PSG': 'Paris SG',
        'St Etienne': 'St Etienne', 'Lyon': 'Lyon'
    }

name_map = normalize_team_names()

# 标准化Kaggle球队名称
df_kaggle['HomeTeam'] = df_kaggle['HomeTeam'].replace(name_map)
df_kaggle['AwayTeam'] = df_kaggle['AwayTeam'].replace(name_map)

# 转换日期格式
df_kaggle['MatchDate'] = pd.to_datetime(df_kaggle['MatchDate'])
df_fd['Date'] = pd.to_datetime(df_fd['Date'], format='%d/%m/%Y', errors='coerce')

# 创建匹配键
print("\n创建匹配键...")
df_kaggle['merge_key'] = (
    df_kaggle['Division'] + '_' + 
    df_kaggle['HomeTeam'] + '_' + 
    df_kaggle['AwayTeam'] + '_' + 
    df_kaggle['MatchDate'].dt.strftime('%Y%m%d')
)

df_fd['merge_key'] = (
    df_fd['Div'] + '_' + 
    df_fd['HomeTeam'] + '_' + 
    df_fd['AwayTeam'] + '_' + 
    df_fd['Date'].dt.strftime('%Y%m%d')
)

# 选择Kaggle关键特征
kaggle_features = ['merge_key', 'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 
                   'Form3Away', 'Form5Away']
df_kaggle_select = df_kaggle[kaggle_features].copy()

# 数据合并（左连接：保留所有Football-Data比赛）
print("合并数据...")
df_merged = df_fd.merge(df_kaggle_select, on='merge_key', how='left', suffixes=('', '_kaggle'))

# 匹配统计
matched = df_merged['HomeElo'].notna().sum()
total = len(df_merged)
match_rate = matched / total * 100

print(f"\n匹配结果:")
print(f"  总比赛数: {total:,}")
print(f"  成功匹配: {matched:,}")
print(f"  匹配率: {match_rate:.1f}%")

# 计算衍生特征
print("\n计算衍生特征...")

# 1. 射正率
df_merged['Home_SoT_Rate'] = np.where(
    df_merged['HS'] > 0, df_merged['HST'] / df_merged['HS'], np.nan
)
df_merged['Away_SoT_Rate'] = np.where(
    df_merged['AS'] > 0, df_merged['AST'] / df_merged['AS'], np.nan
)

# 2. ELO差值
df_merged['ELO_Diff'] = df_merged['HomeElo'] - df_merged['AwayElo']

# 3. Form差值
df_merged['Form3_Diff'] = df_merged['Form3Home'] - df_merged['Form3Away']
df_merged['Form5_Diff'] = df_merged['Form5Home'] - df_merged['Form5Away']

# 4. 博彩赔率隐含概率
if 'B365H' in df_merged.columns:
    df_merged['B365_Prob_H'] = 1 / df_merged['B365H']
    df_merged['B365_Prob_D'] = 1 / df_merged['B365D']
    df_merged['B365_Prob_A'] = 1 / df_merged['B365A']

# 5. 比赛强度（总ELO）
df_merged['Match_Intensity'] = df_merged['HomeElo'] + df_merged['AwayElo']

# 删除merge_key
df_merged = df_merged.drop('merge_key', axis=1)

# 保存合并数据
output_file = os.path.join(data_dir, 'merged_2020_2025.csv')
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(f"合并完成！")
print(f"  最终数据: {len(df_merged):,} 行 × {len(df_merged.columns)} 列")
print(f"  保存到: {output_file}")
print(f"{'='*60}")

# 按赛季统计
print("\n按赛季分布:")
season_counts = df_merged['Season'].value_counts().sort_index()
for season, count in season_counts.items():
    print(f"  {season}: {count:,}")

# 按联赛统计
print("\n按联赛分布:")
league_counts = df_merged['Div'].value_counts()
league_names = {'E0': '英超', 'SP1': '西甲', 'I1': '意甲', 'D1': '德甲', 'F1': '法甲'}
for league, count in league_counts.items():
    print(f"  {league_names.get(league, league)}: {count:,}")

# 特征完整性检查
print("\n关键特征缺失率:")
key_features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 
                'HS', 'AS', 'HST', 'AST', 'B365H']
for feature in key_features:
    if feature in df_merged.columns:
        missing_pct = df_merged[feature].isna().mean() * 100
        print(f"  {feature}: {missing_pct:.1f}%")

# 数据示例
print("\n数据示例:")
print(df_merged[['Season', 'Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 
                 'HomeElo', 'AwayElo', 'Form3Home', 'Form3Away']].head())
