"""
方案1：直接下载Kaggle数据集（Club Football Match Data 2000-2025）
使用Kaggle API自动下载并解压数据集
"""

import os
import sys
import zipfile
import pandas as pd
from pathlib import Path

def check_kaggle_setup():
    """检查Kaggle API配置"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API配置正确")
        return api
    except Exception as e:
        print(f"❌ Kaggle API未配置: {e}")
        print("\n请按以下步骤配置Kaggle API:")
        print("1. 访问 https://www.kaggle.com/settings")
        print("2. 滚动到 'API' 部分，点击 'Create New API Token'")
        print("3. 下载 kaggle.json 文件")
        print("4. 将文件放置到: C:\\Users\\<你的用户名>\\.kaggle\\kaggle.json")
        print("5. 安装kaggle包: pip install kaggle")
        return None

def download_kaggle_dataset():
    """下载Kaggle数据集"""
    # 检查API
    api = check_kaggle_setup()
    if api is None:
        return False
    
    # 设置下载路径
    base_path = Path(__file__).parent / "data"
    base_path.mkdir(exist_ok=True)
    
    dataset_name = "adamgbor/club-football-match-data-2000-2025"
    
    print(f"\n{'='*80}")
    print("开始下载Kaggle数据集")
    print(f"{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"下载路径: {base_path}")
    
    try:
        # 下载并解压数据集
        print("\n正在下载...")
        api.dataset_download_files(dataset_name, path=base_path, unzip=True)
        print("✅ 数据集下载完成")
        
        # 列出下载的文件
        files = list(base_path.glob("*.csv"))
        print(f"\n下载的文件:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def analyze_downloaded_data():
    """分析下载的数据"""
    base_path = Path(__file__).parent / "data"
    
    # 查找matches文件
    matches_files = list(base_path.glob("*matches*.csv")) + list(base_path.glob("*MATCHES*.csv"))
    elo_files = list(base_path.glob("*elo*.csv")) + list(base_path.glob("*ELO*.csv"))
    
    if not matches_files:
        print("\n❌ 未找到matches数据文件")
        return
    
    matches_file = matches_files[0]
    print(f"\n{'='*80}")
    print("数据分析：Kaggle数据集")
    print(f"{'='*80}")
    
    # 读取matches数据
    df = pd.read_csv(matches_file)
    print(f"\n【比赛数据文件】: {matches_file.name}")
    print(f"总比赛数: {len(df):,}")
    print(f"总特征数: {len(df.columns)}")
    
    # 转换日期
    if 'MatchDate' in df.columns:
        df['MatchDate'] = pd.to_datetime(df['MatchDate'])
        
        # 筛选2025-26赛季数据
        df_2526 = df[df['MatchDate'] >= '2025-08-01'].copy()
        print(f"\n2025-26赛季比赛数: {len(df_2526):,}")
        
        if len(df_2526) > 0:
            # 联赛分布
            if 'Division' in df_2526.columns:
                print("\n联赛分布:")
                league_counts = df_2526['Division'].value_counts()
                for league, count in league_counts.head(10).items():
                    print(f"  - {league}: {count}场")
            
            # 日期范围
            print(f"\n日期范围: {df_2526['MatchDate'].min()} 到 {df_2526['MatchDate'].max()}")
    
    # 显示特征列表
    print(f"\n可用特征（前20个）:")
    for i, col in enumerate(df.columns[:20], 1):
        print(f"  {i}. {col}")
    
    if len(df.columns) > 20:
        print(f"  ... 还有 {len(df.columns) - 20} 个特征")
    
    # 关键特征检查
    key_features = ['HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away']
    print(f"\n✅ Kaggle数据集的优势特征:")
    for feat in key_features:
        if feat in df.columns:
            missing_pct = df[feat].isna().sum() / len(df) * 100
            print(f"  ✓ {feat}: 缺失率 {missing_pct:.1f}%")
    
    # 读取ELO数据
    if elo_files:
        elo_file = elo_files[0]
        print(f"\n【ELO评分文件】: {elo_file.name}")
        df_elo = pd.read_csv(elo_file)
        print(f"总记录数: {len(df_elo):,}")
        print(f"覆盖球队数: {df_elo['Club'].nunique() if 'Club' in df_elo.columns else 'N/A'}")

if __name__ == "__main__":
    print("方案1：Kaggle数据集下载")
    print("="*80)
    
    # 下载数据
    success = download_kaggle_dataset()
    
    if success:
        # 分析数据
        analyze_downloaded_data()
        
        print(f"\n{'='*80}")
        print("✅ 方案1数据获取完成！")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("请配置Kaggle API后重新运行此脚本")
        print(f"{'='*80}")
