import numpy as np
import pandas as pd

def check_dph_lag_correlation(
    df,
    targets=("total_dph", "buy_box_dph", "in_stock_dph"),
    asin_col="asin",
    week_col="order_week",
    lags=(1, 2, 4, 8, 13, 26, 52),
    min_weeks=80,
):
    d = df.copy()
    d[asin_col] = d[asin_col].astype(str)
    d[week_col] = pd.to_datetime(d[week_col])
    d = d.sort_values([asin_col, week_col])

    rows = []

    for target in targets:
        d[target] = pd.to_numeric(d[target], errors="coerce").fillna(0.0).clip(lower=0.0)

        for lag in lags:
            corrs = []

            for asin, g in d.groupby(asin_col):
                if len(g) < max(min_weeks, lag + 10):
                    continue

                y = np.log1p(g[target].values.astype(float))
                y1 = y[lag:]
                y0 = y[:-lag]

                if np.std(y1) < 1e-8 or np.std(y0) < 1e-8:
                    continue

                corrs.append(np.corrcoef(y1, y0)[0, 1])

            rows.append({
                "target": target,
                "lag": lag,
                "mean_corr": np.nanmean(corrs) if len(corrs) else np.nan,
                "median_corr": np.nanmedian(corrs) if len(corrs) else np.nan,
                "n_asins": len(corrs),
            })

    out = pd.DataFrame(rows)
    display(out.pivot(index="lag", columns="target", values="median_corr").round(4))
    return out

lag_corr_df = check_dph_lag_correlation(data_raw1)
