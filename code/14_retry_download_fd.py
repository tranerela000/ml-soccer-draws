"""
重试下载失败的Football-Data.co.uk历史数据（增强版）
"""
import pandas as pd
import requests
import os
import time

# 数据路径
data_dir = 'data'

# 已下载的数据
existing_file = os.path.join(data_dir, 'footballdata_uk_2020_2025.csv')
df_existing = pd.read_csv(existing_file)
df_existing.columns = df_existing.columns.str.replace('ï»¿', '')

print(f"已有数据: {len(df_existing)} 场比赛")
print(f"已覆盖赛季: {df_existing['Season'].unique()}")

# 需要补充下载的赛季和联赛
missing_data = [
    ('2223', '2022-23', 'F1', '法甲'),
    ('2324', '2023-24', 'E0', '英超'),
    ('2324', '2023-24', 'SP1', '西甲'),
    ('2324', '2023-24', 'I1', '意甲'),
    ('2324', '2023-24', 'D1', '德甲'),
    ('2324', '2023-24', 'F1', '法甲'),
    ('2425', '2024-25', 'E0', '英超'),
    ('2425', '2024-25', 'SP1', '西甲'),
    ('2425', '2024-25', 'I1', '意甲'),
]

base_url = 'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'

new_data = []
success_count = 0
fail_list = []

print("\n开始重试下载缺失数据...\n")

for season_code, season_label, league_code, league_name in missing_data:
    url = base_url.format(season=season_code, league=league_code)
    
    # 重试机制
    for attempt in range(3):
        try:
            print(f"  [{attempt+1}/3] {season_label} {league_name}({league_code})...", end=' ')
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df.columns = df.columns.str.replace('ï»¿', '')
            
            df['Div'] = league_code
            df['Season'] = season_label
            
            new_data.append(df)
            success_count += 1
            print(f"✅ {len(df)} 场")
            break
            
        except Exception as e:
            if attempt < 2:
                print(f"失败，等待重试...")
                time.sleep(2)
            else:
                print(f"❌ 失败（{type(e).__name__}）")
                fail_list.append(f"{season_label} {league_name}")
        
        time.sleep(1)

# 合并新旧数据
if new_data:
    df_new = pd.concat(new_data, ignore_index=True)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # 保存
    output_file = os.path.join(data_dir, 'footballdata_uk_2020_2025.csv')
    df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"补充下载统计:")
    print(f"  新增成功: {success_count} 个文件")
    print(f"  新增比赛: {len(df_new):,}")
    print(f"  总比赛数: {len(df_combined):,}")
    
    if fail_list:
        print(f"\n仍然失败的文件:")
        for item in fail_list:
            print(f"    - {item}")
    
    print(f"\n按赛季分布:")
    season_counts = df_combined['Season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"  {season}: {count:,}")
    
    print(f"{'='*60}")
else:
    print("\n没有新数据下载成功")
