import numpy as np
import pandas as pd

def quick_check_review_count_leakage(
    df,
    asin_col="asin",
    week_col="order_week",
    review_col="customer_active_review_count",
    history=52,
    horizon=20,
):
    d = df.copy()
    d[asin_col] = d[asin_col].astype(str)
    d[week_col] = pd.to_datetime(d[week_col])
    
    if review_col not in d.columns:
        raise ValueError(f"{review_col} not found in dataframe.")
    
    d[review_col] = pd.to_numeric(d[review_col], errors="coerce")
    d = d.sort_values([asin_col, week_col]).reset_index(drop=True)
    
    rows = []
    
    for asin, g in d.groupby(asin_col):
        g = g.sort_values(week_col).reset_index(drop=True)
        x = g[review_col].fillna(method="ffill").fillna(0.0).values.astype(float)
        
        if len(x) < 2:
            continue
        
        diff = np.diff(x)
        
        rows.append({
            "asin": asin,
            "n_weeks": len(x),
            "start_val": x[0],
            "end_val": x[-1],
            "mean_val": np.mean(x),
            "std_val": np.std(x),
            "n_changes": np.sum(diff != 0),
            "change_rate": np.mean(diff != 0),
            "n_increase": np.sum(diff > 0),
            "n_decrease": np.sum(diff < 0),
            "decrease_rate": np.mean(diff < 0),
            "max_drop": diff.min() if len(diff) > 0 else 0,
            "max_jump": diff.max() if len(diff) > 0 else 0,
        })
    
    stat = pd.DataFrame(rows)
    
    print("=" * 100)
    print(f"Review count quick check: {review_col}")
    print("=" * 100)
    print("ASINs checked:", stat["asin"].nunique())
    print("Average change_rate:", stat["change_rate"].mean())
    print("Median change_rate:", stat["change_rate"].median())
    print("Average decrease_rate:", stat["decrease_rate"].mean())
    print("Median decrease_rate:", stat["decrease_rate"].median())
    print("Share of ASINs with any decrease:", (stat["n_decrease"] > 0).mean())
    print("Share of ASINs never change:", (stat["n_changes"] == 0).mean())
    
    print("\nTop ASINs with most decreases:")
    display(
        stat.sort_values("n_decrease", ascending=False)
            .head(20)
            [["asin", "n_weeks", "start_val", "end_val", "n_changes", "change_rate", "n_decrease", "decrease_rate", "max_drop", "max_jump"]]
    )
    
    # 检查最后一个 validation window 里，future review_count 是否相对 forecast origin 有变化
    leak_rows = []
    
    for asin, g in d.groupby(asin_col):
        g = g.sort_values(week_col).reset_index(drop=True)
        
        if len(g) < history + horizon:
            continue
        
        origin_idx = len(g) - horizon - 1
        future_start = len(g) - horizon
        future_end = len(g)
        
        origin_val = g.loc[origin_idx, review_col]
        future_vals = g.loc[future_start:future_end-1, review_col].fillna(origin_val).values.astype(float)
        
        changed_future = np.any(np.abs(future_vals - origin_val) > 1e-8)
        future_change_rate = np.mean(np.abs(future_vals - origin_val) > 1e-8)
        
        leak_rows.append({
            "asin": asin,
            "origin_week": g.loc[origin_idx, week_col],
            "origin_val": origin_val,
            "future_min": np.nanmin(future_vals),
            "future_max": np.nanmax(future_vals),
            "future_mean": np.nanmean(future_vals),
            "future_changed_from_origin": changed_future,
            "future_change_rate": future_change_rate,
        })
    
    leak_df = pd.DataFrame(leak_rows)
    
    print("\n" + "=" * 100)
    print("Future-window leakage risk check")
    print("=" * 100)
    print("ASINs checked:", leak_df["asin"].nunique())
    print("Share future review_count changes from forecast origin:", leak_df["future_changed_from_origin"].mean())
    print("Average future_change_rate:", leak_df["future_change_rate"].mean())
    
    print("\nExamples where future review_count changes:")
    display(
        leak_df[leak_df["future_changed_from_origin"]]
            .sort_values("future_change_rate", ascending=False)
            .head(20)
    )
    
    return stat, leak_df


review_stat, review_future_check = quick_check_review_count_leakage(
    data_raw1,
    asin_col="asin",
    week_col="order_week",
    review_col="customer_active_review_count",
    history=52,
    horizon=20,
)
