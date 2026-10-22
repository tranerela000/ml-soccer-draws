"""
严格时间序列分离的特征工程（优化版）
关键修复：训练集和测试集完全独立计算特征，避免信息泄漏
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

def print_flush(msg):
    """立即刷新输出"""
    print(msg)
    sys.stdout.flush()

print_flush("=" * 70)
print_flush("严格分离的特征工程（优化版）")
print_flush("=" * 70)

# 加载数据
print_flush("\n[1/5] 加载数据...")
train_df = pd.read_csv('data/merged_2020_2025.csv')
test_df = pd.read_csv('data/test_set_2025_26.csv')

train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)

print_flush(f"训练集: {len(train_df)} 场比赛 (截止 {train_df['Date'].max().date()})")
print_flush(f"测试集: {len(test_df)} 场比赛 (起始 {test_df['Date'].min().date()})")

# 验证时间分离
assert train_df['Date'].max() < test_df['Date'].min(), "训练集和测试集时间重叠！"
print_flush("✓ 训练集和测试集时间严格分离")

# ============================================================
# 特征1: H2H对阵记录（优化版）
# ============================================================
print_flush("\n[2/5] 计算H2H特征...")

def calculate_h2h_optimized(df, historical_df):
    """优化的H2H计算"""
    h2h_features = []
    total = len(df)
    
    for idx in range(total):
        if idx % 1000 == 0:
            print_flush(f"  H2H进度: {idx}/{total} ({idx*100//total}%)")
        
        row = df.iloc[idx]
        home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
        
        # 查找历史对阵
        mask = (historical_df['Date'] < date) & (
            ((historical_df['HomeTeam'] == home) & (historical_df['AwayTeam'] == away)) |
            ((historical_df['HomeTeam'] == away) & (historical_df['AwayTeam'] == home))
        )
        historical = historical_df[mask]
        
        if len(historical) == 0:
            h2h_features.append({
                'H2H_Matches': 0,
                'H2H_HomeWinRate': 0.5,
                'H2H_DrawRate': 0.25,
                'H2H_HomeGoals_Avg': 1.5,
                'H2H_AwayGoals_Avg': 1.5
            })
        else:
            # 计算主队视角的统计
            is_home = (historical['HomeTeam'] == home)
            home_wins = ((historical['FTR'] == 'H') & is_home).sum() + \
                       ((historical['FTR'] == 'A') & ~is_home).sum()
            draws = (historical['FTR'] == 'D').sum()
            
            home_goals = np.where(is_home, historical['FTHG'], historical['FTAG'])
            away_goals = np.where(is_home, historical['FTAG'], historical['FTHG'])
            
            total_matches = len(historical)
            h2h_features.append({
                'H2H_Matches': total_matches,
                'H2H_HomeWinRate': home_wins / total_matches,
                'H2H_DrawRate': draws / total_matches,
                'H2H_HomeGoals_Avg': home_goals.mean(),
                'H2H_AwayGoals_Avg': away_goals.mean()
            })
    
    return pd.DataFrame(h2h_features)

# 训练集H2H：使用训练集自身历史
print_flush("\n训练集H2H计算（使用训练集历史）...")
train_h2h = calculate_h2h_optimized(train_df, train_df)
print_flush(f"✓ 训练集H2H完成: {train_h2h.shape}")

# 测试集H2H：只使用训练集历史
print_flush("\n测试集H2H计算（只使用训练集历史）...")
test_h2h = calculate_h2h_optimized(test_df, train_df)
print_flush(f"✓ 测试集H2H完成: {test_h2h.shape}")

# ============================================================
# 特征2: 滚动统计（优化版）
# ============================================================
print_flush("\n[3/5] 计算滚动统计特征...")

def calculate_rolling_stats_optimized(df, historical_df, windows=[5, 10]):
    """优化的滚动统计计算"""
    all_features = []
    total = len(df)
    
    for idx in range(total):
        if idx % 1000 == 0:
            print_flush(f"  滚动统计进度: {idx}/{total} ({idx*100//total}%)")
        
        row = df.iloc[idx]
        home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
        
        features = {}
        
        for window in windows:
            # 主队历史
            home_mask = (historical_df['Date'] < date) & (
                (historical_df['HomeTeam'] == home) | (historical_df['AwayTeam'] == home)
            )
            home_history = historical_df[home_mask].tail(window)
            
            # 客队历史
            away_mask = (historical_df['Date'] < date) & (
                (historical_df['HomeTeam'] == away) | (historical_df['AwayTeam'] == away)
            )
            away_history = historical_df[away_mask].tail(window)
            
            # 主队统计（向量化）
            if len(home_history) > 0:
                is_home_team = (home_history['HomeTeam'] == home)
                home_goals = np.where(is_home_team, home_history['FTHG'], home_history['FTAG'])
                home_conceded = np.where(is_home_team, home_history['FTAG'], home_history['FTHG'])
                home_wins = ((home_history['FTR'] == 'H') & is_home_team).sum() + \
                           ((home_history['FTR'] == 'A') & ~is_home_team).sum()
                
                features[f'Home_L{window}_GoalsScored'] = home_goals.mean()
                features[f'Home_L{window}_GoalsConceded'] = home_conceded.mean()
                features[f'Home_L{window}_WinRate'] = home_wins / len(home_history)
            else:
                features[f'Home_L{window}_GoalsScored'] = 1.5
                features[f'Home_L{window}_GoalsConceded'] = 1.5
                features[f'Home_L{window}_WinRate'] = 0.33
            
            # 客队统计（向量化）
            if len(away_history) > 0:
                is_home_team = (away_history['HomeTeam'] == away)
                away_goals = np.where(is_home_team, away_history['FTHG'], away_history['FTAG'])
                away_conceded = np.where(is_home_team, away_history['FTAG'], away_history['FTHG'])
                away_wins = ((away_history['FTR'] == 'H') & is_home_team).sum() + \
                           ((away_history['FTR'] == 'A') & ~is_home_team).sum()
                
                features[f'Away_L{window}_GoalsScored'] = away_goals.mean()
                features[f'Away_L{window}_GoalsConceded'] = away_conceded.mean()
                features[f'Away_L{window}_WinRate'] = away_wins / len(away_history)
            else:
                features[f'Away_L{window}_GoalsScored'] = 1.5
                features[f'Away_L{window}_GoalsConceded'] = 1.5
                features[f'Away_L{window}_WinRate'] = 0.33
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)

# 训练集滚动统计
print_flush("\n训练集滚动统计（使用训练集历史）...")
train_rolling = calculate_rolling_stats_optimized(train_df, train_df)
print_flush(f"✓ 训练集滚动统计完成: {train_rolling.shape}")

# 测试集滚动统计
print_flush("\n测试集滚动统计（只使用训练集历史）...")
test_rolling = calculate_rolling_stats_optimized(test_df, train_df)
print_flush(f"✓ 测试集滚动统计完成: {test_rolling.shape}")

# ============================================================
# 特征3: 主客场分离统计（优化版）
# ============================================================
print_flush("\n[4/5] 计算主客场分离特征...")

def calculate_home_away_split_optimized(df, historical_df):
    """优化的主客场分离统计"""
    features = []
    total = len(df)
    
    for idx in range(total):
        if idx % 1000 == 0:
            print_flush(f"  主客场进度: {idx}/{total} ({idx*100//total}%)")
        
        row = df.iloc[idx]
        home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
        
        # 主队主场历史
        home_at_home = historical_df[
            (historical_df['Date'] < date) & 
            (historical_df['HomeTeam'] == home)
        ]
        
        # 客队客场历史
        away_at_away = historical_df[
            (historical_df['Date'] < date) & 
            (historical_df['AwayTeam'] == away)
        ]
        
        feat = {}
        
        if len(home_at_home) > 0:
            feat['Home_AtHome_WinRate'] = (home_at_home['FTR'] == 'H').sum() / len(home_at_home)
            feat['Home_AtHome_Goals'] = home_at_home['FTHG'].mean()
        else:
            feat['Home_AtHome_WinRate'] = 0.45
            feat['Home_AtHome_Goals'] = 1.6
        
        if len(away_at_away) > 0:
            feat['Away_AtAway_WinRate'] = (away_at_away['FTR'] == 'A').sum() / len(away_at_away)
            feat['Away_AtAway_Goals'] = away_at_away['FTAG'].mean()
        else:
            feat['Away_AtAway_WinRate'] = 0.30
            feat['Away_AtAway_Goals'] = 1.3
        
        features.append(feat)
    
    return pd.DataFrame(features)

print_flush("\n训练集主客场特征（使用训练集历史）...")
train_home_away = calculate_home_away_split_optimized(train_df, train_df)
print_flush(f"✓ 训练集主客场完成: {train_home_away.shape}")

print_flush("\n测试集主客场特征（只使用训练集历史）...")
test_home_away = calculate_home_away_split_optimized(test_df, train_df)
print_flush(f"✓ 测试集主客场完成: {test_home_away.shape}")

# ============================================================
# 合并所有特征
# ============================================================
print_flush("\n[5/5] 合并特征...")

train_final = pd.concat([
    train_df.reset_index(drop=True),
    train_h2h.reset_index(drop=True),
    train_rolling.reset_index(drop=True),
    train_home_away.reset_index(drop=True)
], axis=1)

test_final = pd.concat([
    test_df.reset_index(drop=True),
    test_h2h.reset_index(drop=True),
    test_rolling.reset_index(drop=True),
    test_home_away.reset_index(drop=True)
], axis=1)

print_flush(f"\n训练集最终: {len(train_final)} 场, {len(train_final.columns)} 列")
print_flush(f"测试集最终: {len(test_final)} 场, {len(test_final.columns)} 列")

# 保存
train_final.to_csv('data/processed/train_strict.csv', index=False, encoding='utf-8-sig')
test_final.to_csv('data/processed/test_strict.csv', index=False, encoding='utf-8-sig')

print_flush("\n✓ 严格分离的数据已保存:")
print_flush("  - data/processed/train_strict.csv")
print_flush("  - data/processed/test_strict.csv")

# 验证
print_flush("\n" + "=" * 70)
print_flush("严格分离验证")
print_flush("=" * 70)

print_flush("\n新增特征:")
new_features = [col for col in train_final.columns if col not in train_df.columns]
print_flush(f"  数量: {len(new_features)}")
print_flush(f"  列表: {new_features}")

print_flush("\n✓ 特征工程完成，确保严格时间序列分离")
