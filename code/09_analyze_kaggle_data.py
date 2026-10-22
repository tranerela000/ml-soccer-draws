"""
分析已下载的Kaggle数据集（方案1）
运行前提：已手动下载并解压数据到data目录
"""

import pandas as pd
from pathlib import Path
import numpy as np

def find_kaggle_files():
    """查找Kaggle数据文件"""
    data_dir = Path(__file__).parent / "data"
    
    # 查找matches文件
    matches_patterns = ["*MATCHES*.csv", "*matches*.csv", "*match*.csv"]
    matches_file = None
    for pattern in matches_patterns:
        files = list(data_dir.glob(pattern))
        if files:
            matches_file = files[0]
            break
    
    # 查找ELO文件
    elo_patterns = ["*ELO*.csv", "*elo*.csv"]
    elo_file = None
    for pattern in elo_patterns:
        files = list(data_dir.glob(pattern))
        if files:
            elo_file = files[0]
            break
    
    return matches_file, elo_file

def analyze_kaggle_data():
    """分析Kaggle数据集"""
    matches_file, elo_file = find_kaggle_files()
    
    if not matches_file:
        print("❌ 未找到MATCHES.csv文件")
        print("\n请先运行: python 08_kaggle_manual_guide.py")
        print("按照指引下载数据后再运行此脚本")
        return
    
    print("="*80)
    print("【方案1】Kaggle数据集分析")
    print("="*80)
    
    # 读取matches数据
    print(f"\n正在读取: {matches_file.name}")
    df = pd.read_csv(matches_file)
    
    print(f"\n总比赛数: {len(df):,}")
    print(f"总特征数: {len(df.columns)}")
    
    # 转换日期
    if 'MatchDate' in df.columns:
        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
        print(f"日期范围: {df['MatchDate'].min()} 到 {df['MatchDate'].max()}")
        
        # 筛选2025-26赛季
        df_2526 = df[df['MatchDate'] >= '2025-08-01'].copy()
        print(f"\n2025-26赛季比赛数: {len(df_2526):,}")
        
        if len(df_2526) > 0:
            # 联赛分布
            if 'Division' in df_2526.columns:
                print("\n五大联赛分布:")
                five_leagues = ['E0', 'SP1', 'I1', 'D1', 'F1']
                league_names = {
                    'E0': '英超',
                    'SP1': '西甲',
                    'I1': '意甲',
                    'D1': '德甲',
                    'F1': '法甲'
                }
                
                total_five = 0
                for league in five_leagues:
                    count = len(df_2526[df_2526['Division'] == league])
                    if count > 0:
                        print(f"  - {league_names[league]} ({league}): {count}场")
                        total_five += count
                
                print(f"\n五大联赛总计: {total_five}场")
                
                other_count = len(df_2526) - total_five
                if other_count > 0:
                    print(f"其他联赛: {other_count}场")
    
    # 关键特征分析
    print("\n" + "="*80)
    print("方案1独有的优势特征")
    print("="*80)
    
    key_features = {
        'HomeElo': 'ELO评分（主队）',
        'AwayElo': 'ELO评分（客队）',
        'Form3Home': '近3场战绩（主队）',
        'Form5Home': '近5场战绩（主队）',
        'Form3Away': '近3场战绩（客队）',
        'Form5Away': '近5场战绩（客队）'
    }
    
    available_features = []
    for feat, desc in key_features.items():
        if feat in df.columns:
            if len(df_2526) > 0:
                missing_pct = df_2526[feat].isna().sum() / len(df_2526) * 100
                completeness = 100 - missing_pct
                print(f"✅ {feat:15} - {desc:20} (完整性: {completeness:.1f}%)")
                available_features.append(feat)
            else:
                print(f"✅ {feat:15} - {desc}")
                available_features.append(feat)
        else:
            print(f"❌ {feat:15} - {desc} (未找到)")
    
    # 数据示例
    if len(df_2526) > 0:
        print("\n" + "="*80)
        print("2025-26赛季数据示例（最近5场）")
        print("="*80)
        
        display_cols = ['MatchDate', 'Division', 'HomeTeam', 'AwayTeam']
        if 'FTHG' in df_2526.columns and 'FTAG' in df_2526.columns:
            display_cols += ['FTHG', 'FTAG']
        
        for feat in available_features[:4]:
            if feat in df_2526.columns:
                display_cols.append(feat)
        
        print(df_2526[display_cols].tail(5).to_string(index=False))
    
    # ELO评分数据
    if elo_file:
        print("\n" + "="*80)
        print("ELO评分数据")
        print("="*80)
        print(f"文件: {elo_file.name}")
        
        df_elo = pd.read_csv(elo_file)
        print(f"总记录数: {len(df_elo):,}")
        
        if 'Club' in df_elo.columns:
            print(f"覆盖球队数: {df_elo['Club'].nunique()}")
        
        if 'Date' in df_elo.columns:
            df_elo['Date'] = pd.to_datetime(df_elo['Date'], errors='coerce')
            print(f"时间跨度: {df_elo['Date'].min()} 到 {df_elo['Date'].max()}")
        
        # 显示示例
        print("\nELO评分示例（最新记录）:")
        display_cols_elo = [col for col in ['Date', 'Club', 'Country', 'Elo'] if col in df_elo.columns]
        print(df_elo[display_cols_elo].tail(5).to_string(index=False))
    
    # 保存2525数据
    if len(df_2526) > 0:
        output_file = Path(__file__).parent / "data" / "kaggle_2025_26.csv"
        df_2526.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 2025-26赛季数据已保存到: {output_file.name}")
    
    print("\n" + "="*80)
    print("✅ 方案1数据分析完成")
    print("="*80)
    print("\n下一步：运行对比分析脚本")
    print("  python 10_compare_two_solutions.py")

if __name__ == "__main__":
    analyze_kaggle_data()
