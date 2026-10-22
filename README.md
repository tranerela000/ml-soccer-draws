# Why Machine Learning Models Almost Never Predict Draws in Soccer

Code and data for the study: *Why machine learning models almost never predict draws in soccer: A multi-algorithm, multi-league, multi-season empirical study*.

Machine learning models for soccer match outcome prediction rarely predict draws, even though roughly one in four matches ends in a draw. This repository contains the full pipeline used to examine why: data download, feature engineering, model training (XGBoost, LightGBM, random forest, logistic regression, and an XGBoost + LSTM hybrid), and all analyses reported in the paper (draw analysis, odds-gap analysis, cross-league validation, multi-algorithm validation, time trends, subset analyses, and McNemar tests).

## Data sources

| Source | Content | License |
|---|---|---|
| [Football-Data.co.uk](https://www.football-data.co.uk) | Match results, bookmaker odds, match statistics (5 major European leagues, 2020–2026) | Free for research use; cite the source |
| Kaggle: [`adamgbor/club-football-match-data-2000-2025`](https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025) | Elo ratings and form features used for team-strength variables | See the dataset page on Kaggle |

The `data/` folder contains the exact files used in the paper. Alternatively, you can regenerate them by re-running the download scripts.

## Repository structure

```
├── code/                 # All scripts, run in the order below
├── data/                 # Input data (raw and merged)
│   └── processed/        # Final training/test sets
├── models/               # Final trained LSTM and stacking outputs
└── results/              # Result tables and figures underlying the paper
```

## Pipeline

Run from the repository root (`python code/<script>.py`):

1. `13_download_historical_fd.py` — download Football-Data.co.uk match data 2020–2025
2. `07_download_kaggle_direct.py` — download the Kaggle dataset (requires a free Kaggle API token in `~/.kaggle/kaggle.json`)
3. `12_analyze_historical_kaggle.py` and `09_analyze_kaggle_data.py` — extract the 2020–2025 and 2025–26 subsets from the Kaggle package (their outputs `kaggle_2020_2025.csv` and `kaggle_2025_26.csv` are already provided in `data/`)
4. `15_merge_historical_data.py` — merge the two training sources into `data/merged_2020_2025.csv` (about 90% of matches receive Elo/form values; the rest use only Football-Data features)
5. `16_download_full_2025_26.py` — download the 2025–26 season from Football-Data.co.uk
6. `17_prepare_test_set.py` — merge the test set with the Kaggle Elo/form features into `data/test_set_2025_26.csv`
7. `30_strict_feature_engineering_optimized.py` — build strictly pre-match features (H2H, rolling form, home/away profiles)
8. `32_tier1_features_and_ablation.py` — feature-family ablation experiments
9. `34_hist_result_features.py` — historical-result features for the LSTM branch
10. `35_hybrid_retrain_clean.py` — train the XGBoost + LSTM hybrid model
11. Analyses:
   - `33_odds_gap_analysis.py` — draw probability vs. odds gap
   - `34_draw_analysis.py` — draw miss rates and confidence distributions
   - `36_cross_league_validation.py` — per-league and pooled validation
   - `37_multi_algorithm_validation.py` — five algorithms compared
   - `38_time_trend_analysis.py` — draw rates and model behavior by season
   - `39_subset_analysis.py` — strength-gap and schedule-congestion subsets
   - `41_mcnemar_test.py` — McNemar tests vs. the bookmaker-favorite benchmark
10. `40_paper_figures.py` — generate all paper figures (`results/figures/`)

Trained artifacts (`models/lstm_clean.keras`, `models/stack_oof_probs.npy`) and the final data tables are included so the analyses can be inspected without retraining.

## Environment

Python 3.11+ (developed on 3.14). Install dependencies:

```bash
pip install -r requirements.txt
```

TensorFlow is only required for the hybrid model (step 8); all other steps run without it.

## License

Code is released under the MIT License (see `LICENSE`). The data files originate from Football-Data.co.uk and Kaggle and remain subject to the terms of those sources.

## Citation

If you use this code or data, please cite the paper (DOI to be added upon publication) and the original data sources.

## Author

Jiaxiong Li — Wuhan City Polytechnic, Wuhan, China
ORCID: [0009-0001-9873-4728](https://orcid.org/0009-0001-9873-4728)
