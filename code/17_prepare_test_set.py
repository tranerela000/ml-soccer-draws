"""
合并完整2025-26赛季测试集：Football-Data + Kaggle ELO/Form
"""
import pandas as pd
import numpy as np
import os

data_dir = 'data'

# 加载数据
print("正在加载数据...")
df_fd = pd.read_csv(os.path.join(data_dir, 'footballdata_uk_2025_26_full.csv'), low_memory=False)
df_kaggle = pd.read_csv(os.path.join(data_dir, 'kaggle_2025_26.csv'))

# 清理BOM
df_fd.columns = df_fd.columns.str.replace('ï»¿', '')
df_kaggle.columns = df_kaggle.columns.str.replace('ï»¿', '')

print(f"Football-Data: {len(df_fd):,} 场")
print(f"Kaggle: {len(df_kaggle):,} 场")

# 球队名称标准化
name_map = {
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

df_kaggle['HomeTeam'] = df_kaggle['HomeTeam'].replace(name_map)
df_kaggle['AwayTeam'] = df_kaggle['AwayTeam'].replace(name_map)

# 转换日期
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

# 筛选五大联赛Kaggle数据
df_kaggle_5 = df_kaggle[df_kaggle['Division'].isin(['E0', 'SP1', 'I1', 'D1', 'F1'])].copy()
print(f"Kaggle五大联赛: {len(df_kaggle_5):,} 场")

# 选择Kaggle关键特征
kaggle_features = ['merge_key', 'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 
                   'Form3Away', 'Form5Away']
df_kaggle_select = df_kaggle_5[kaggle_features].copy()

# 合并数据
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

# 5. 比赛强度
df_merged['Match_Intensity'] = df_merged['HomeElo'] + df_merged['AwayElo']

# 删除merge_key
df_merged = df_merged.drop('merge_key', axis=1)

# 保存
output_file = os.path.join(data_dir, 'test_set_2025_26.csv')
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n{'='*60}")
print(f"测试集生成完成！")
print(f"  数据: {len(df_merged):,} 行 × {len(df_merged.columns)} 列")
print(f"  保存到: test_set_2025_26.csv")
print(f"{'='*60}")

# 按联赛统计
print("\n按联赛分布:")
league_names = {'E0': '英超', 'SP1': '西甲', 'I1': '意甲', 'D1': '德甲', 'F1': '法甲'}
for league, count in df_merged['Div'].value_counts().items():
    print(f"  {league_names.get(league, league)}: {count}")

# 特征完整性
print("\n关键特征缺失率:")
key_features = ['HomeElo', 'AwayElo', 'Form3Home', 'HS', 'AS', 'B365H']
for feature in key_features:
    if feature in df_merged.columns:
        missing_pct = df_merged[feature].isna().mean() * 100
        print(f"  {feature}: {missing_pct:.1f}%")

print("\n数据示例:")
print(df_merged[['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 
                 'HomeElo', 'AwayElo', 'Form3Home']].head())
