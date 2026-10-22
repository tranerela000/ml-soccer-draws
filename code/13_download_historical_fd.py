"""
下载Football-Data.co.uk 2020-2025年五大联赛历史数据
"""
import pandas as pd
import requests
import os
import time

# 数据路径
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# 联赛代码映射
leagues = {
    'E0': '英超',
    'SP1': '西甲',
    'I1': '意甲',
    'D1': '德甲',
    'F1': '法甲'
}

# 赛季列表（格式：yy/yy）
seasons = ['2021', '2122', '2223', '2324', '2425']
season_labels = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

# 基础URL
base_urls = {
    'E0': 'https://www.football-data.co.uk/mmz4281/{season}/E0.csv',
    'SP1': 'https://www.football-data.co.uk/mmz4281/{season}/SP1.csv',
    'I1': 'https://www.football-data.co.uk/mmz4281/{season}/I1.csv',
    'D1': 'https://www.football-data.co.uk/mmz4281/{season}/D1.csv',
    'F1': 'https://www.football-data.co.uk/mmz4281/{season}/F1.csv'
}

all_data = []
success_count = 0
fail_count = 0

print("开始下载Football-Data.co.uk历史数据...\n")

for season_code, season_label in zip(seasons, season_labels):
    print(f"正在下载 {season_label} 赛季数据...")
    
    for league_code, league_name in leagues.items():
        url = base_urls[league_code].format(season=season_code)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # 解析CSV数据
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # 清理BOM字符
            df.columns = df.columns.str.replace('ï»¿', '')
            
            # 添加联赛代码和赛季标签
            df['Div'] = league_code
            df['Season'] = season_label
            
            all_data.append(df)
            success_count += 1
            print(f"  ✅ {league_name}({league_code}): {len(df)} 场比赛")
            
        except Exception as e:
            fail_count += 1
            print(f"  ❌ {league_name}({league_code}): 下载失败 - {str(e)}")
        
        # 避免请求过快
        time.sleep(0.5)
    
    print()

# 合并所有数据
if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 保存到文件
    output_file = os.path.join(data_dir, 'footballdata_uk_2020_2025.csv')
    df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"下载完成统计:")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {fail_count} 个文件")
    print(f"  总比赛数: {len(df_all):,}")
    print(f"  保存到: {output_file}")
    print(f"{'='*60}")
    
    # 按联赛统计
    print("\n按联赛分布:")
    league_counts = df_all['Div'].value_counts()
    for league, count in league_counts.items():
        print(f"  {leagues.get(league, league)}: {count:,}")
    
    # 按赛季统计
    print("\n按赛季分布:")
    season_counts = df_all['Season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"  {season}: {count:,}")
    
    # 特征统计
    print(f"\n总特征数: {len(df_all.columns)}")
    print(f"数据示例:")
    print(df_all[['Season', 'Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].head())
    
else:
    print("\n❌ 没有成功下载任何数据！")
