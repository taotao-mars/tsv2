def check_history_window_baselines(
    df,
    target="in_stock_dph",
    asin_col="asin",
    week_col="order_week",
    horizon=20,
    windows=(4, 8, 13, 26, 52),
):
    d = df.copy()
    d[asin_col] = d[asin_col].astype(str)
    d[week_col] = pd.to_datetime(d[week_col])
    d[target] = pd.to_numeric(d[target], errors="coerce").fillna(0.0).clip(lower=0.0)
    d = d.sort_values([asin_col, week_col])

    rows = []

    for w in windows:
        y_all = []
        p_all = []

        for asin, g in d.groupby(asin_col):
            if len(g) < w + horizon:
                continue

            y = g[target].values.astype(float)
            hist = y[-horizon-w:-horizon]
            true = y[-horizon:]

            pred = np.repeat(np.mean(hist), horizon)

            y_all.append(true)
            p_all.append(pred)

        y_all = np.concatenate(y_all)
        p_all = np.concatenate(p_all)

        rows.append({
            "window": w,
            "true_mean": y_all.mean(),
            "pred_mean": p_all.mean(),
            "ratio": p_all.mean() / (y_all.mean() + 1e-8),
            "WAPE": np.abs(y_all - p_all).sum() / (np.abs(y_all).sum() + 1e-8),
            "corr": np.corrcoef(y_all, p_all)[0, 1] if np.std(p_all) > 1e-8 and np.std(y_all) > 1e-8 else np.nan,
        })

    out = pd.DataFrame(rows)
    display(out.round(4))
    return out

baseline_window_df = check_history_window_baselines(data_raw1, target="in_stock_dph")
