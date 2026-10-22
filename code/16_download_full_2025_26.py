"""
重新下载完整的2025-26赛季数据（2026年10月）
"""
import pandas as pd
import requests
import os
import time
from io import StringIO

data_dir = 'data'

# 联赛代码
leagues = {
    'E0': '英超',
    'SP1': '西甲', 
    'I1': '意甲',
    'D1': '德甲',
    'F1': '法甲'
}

base_url = 'https://www.football-data.co.uk/mmz4281/2526/{league}.csv'

all_data = []
success_count = 0

print("正在下载完整2025-26赛季数据...\n")

for league_code, league_name in leagues.items():
    url = base_url.format(league=league_code)
    
    for attempt in range(3):
        try:
            print(f"  [{attempt+1}/3] {league_name}({league_code})...", end=' ')
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            df.columns = df.columns.str.replace('ï»¿', '')
            
            df['Div'] = league_code
            all_data.append(df)
            success_count += 1
            
            print(f"✅ {len(df)} 场比赛")
            break
            
        except Exception as e:
            if attempt < 2:
                print(f"失败，重试...")
                time.sleep(2)
            else:
                print(f"❌ 失败 - {type(e).__name__}")
        
        time.sleep(1)

if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    
    output_file = os.path.join(data_dir, 'footballdata_uk_2025_26_full.csv')
    df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print(f"下载完成！")
    print(f"  成功: {success_count}/5 个联赛")
    print(f"  总比赛数: {len(df_all):,}")
    print(f"  保存到: footballdata_uk_2025_26_full.csv")
    print(f"{'='*60}")
    
    print("\n按联赛分布:")
    for league, count in df_all['Div'].value_counts().items():
        print(f"  {leagues.get(league, league)}: {count}")
    
    print("\n前5场比赛:")
    print(df_all[['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].head())
    
    # 检查是否完整赛季
    expected = {'E0': 380, 'SP1': 380, 'I1': 380, 'D1': 306, 'F1': 306}
    print("\n赛季完整性检查:")
    for league_code in leagues.keys():
        actual = len(df_all[df_all['Div'] == league_code])
        exp = expected.get(league_code, 0)
        status = "✅ 完整" if actual >= exp * 0.95 else "⚠️ 不完整"
        print(f"  {leagues[league_code]}: {actual}/{exp} 场 {status}")
else:
    print("\n❌ 下载失败")
