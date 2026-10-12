# Data

No data files are committed. The notebook is saved **with its original outputs**, so all results can be read on GitHub without running it. To re-run it, place the files below in the **same folder as the notebook** (the notebook reads them from its working directory). They are excluded by `.gitignore`.

| File | Content | Source |
|------|---------|--------|
| `crsp_sp500_2007_2023.csv.gz` (or `crsp/crsp_sp500_2007_2023.csv`) | CRSP daily file, S&P 500 constituents, 2007–2023 (CIZ format) | WRDS, licensed; may not be redistributed |
| `Data_PCLab4_Stock_Price.csv` | Daily close of AAPL, BA, T, MGM, AMZN, IBM, TSLA, GOOG and `sp500`, 12 Jan 2012 – 11 Aug 2020 | Course-provided |
| `Data_PCLab4_Stock_Volume.csv` | Daily volume for the same securities and dates | Course-provided |

**CRSP fields used:**

| CRSP field | Name in notebook | Use |
|------------|-----------------|-----|
| `DlyCalDt` | `date` | Trading date |
| `PERMNO` | `permno` | Security identifier |
| `PERMCO` | `permco` | Company identifier (firm size across share classes) |
| `Ticker` | `ticker` | Labels |
| `DlyPrc` | `price` | Price |
| `DlyRet` | `ret` | Total return |
| `DlyCumFacPr` | `adj_factor` | Price adjustment factor |
| `DlyCap` | `mcap` | Market capitalisation |
| `DlyPrcFlg` | `price_flag` | Trade / delisting flags |
| `DlyVol` | `vol` | Volume |
| `DlyCumFacShr` | `fac_shr` | Share adjustment factor (split-adjusted volume) |
| `ShrOut` | `shrout` | Shares outstanding |

**Cache.** On first run the notebook writes `cache/*.joblib` (fitted models and intermediate tables) and `exports/*.pkl` (the engineered CRSP panel and predictions). Both contain CRSP-derived data and must never be committed.
