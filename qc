import numpy as np
import pandas as pd
from IPython.display import display

def attach_gl_info(df, data_raw1):
    out = df.copy()
    out["asin"] = out["asin"].astype(str)

    meta_cols = ["asin"]
    for c in ["gl_product_group", "gl_product_group_desc", "category_code"]:
        if c in data_raw1.columns:
            meta_cols.append(c)

    meta = data_raw1[meta_cols].drop_duplicates("asin").copy()
    meta["asin"] = meta["asin"].astype(str)

    out = out.merge(meta, on="asin", how="left")
    return out


def _safe_wape(y, yhat):
    denom = np.abs(y).sum()
    if denom <= 1e-8:
        return np.nan
    return np.abs(yhat - y).sum() / denom


def _quantile_penalty(y, yhat, q):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    err = y - yhat
    loss = np.where(err >= 0, q * err, (1 - q) * (-err))
    denom = np.abs(y).sum()
    if denom <= 1e-8:
        return np.nan
    return loss.sum() / denom


def _under_over(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    denom = np.abs(y).sum()
    if denom <= 1e-8:
        return np.nan, np.nan
    under = np.maximum(y - yhat, 0).sum() / denom
    over = np.maximum(yhat - y, 0).sum() / denom
    return under, over


def demand_gl_diagnostic(
    demand_pred_df,
    data_raw1,
    true_col="fbi_demand",
    amxl_p50_col="p50_amxl",
    amxl_p70_col="p70_amxl",
    scot_p50_col="p50_scot",
    scot_p70_col="p70_scot",
    min_asins=20,
):
    df = attach_gl_info(demand_pred_df, data_raw1)

    required = [true_col, amxl_p50_col, amxl_p70_col, scot_p50_col, scot_p70_col, "gl_product_group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Missing columns:", missing)
        print("Available columns:", df.columns.tolist())
        return None

    rows = []

    group_cols = ["gl_product_group"]
    if "gl_product_group_desc" in df.columns:
        group_cols.append("gl_product_group_desc")

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n_asins = g["asin"].nunique()
        n_rows = len(g)
        if n_asins < min_asins:
            continue

        y = g[true_col].fillna(0).values

        p50_amxl = g[amxl_p50_col].fillna(0).values
        p70_amxl = g[amxl_p70_col].fillna(0).values
        p50_scot = g[scot_p50_col].fillna(0).values
        p70_scot = g[scot_p70_col].fillna(0).values

        p50_amxl_wape = _safe_wape(y, p50_amxl)
        p50_scot_wape = _safe_wape(y, p50_scot)
        p70_amxl_wape = _safe_wape(y, p70_amxl)
        p70_scot_wape = _safe_wape(y, p70_scot)

        p50_amxl_pen = _quantile_penalty(y, p50_amxl, q=0.5)
        p50_scot_pen = _quantile_penalty(y, p50_scot, q=0.5)
        p70_amxl_pen = _quantile_penalty(y, p70_amxl, q=0.7)
        p70_scot_pen = _quantile_penalty(y, p70_scot, q=0.7)

        p50_under, p50_over = _under_over(y, p50_amxl)
        p70_under, p70_over = _under_over(y, p70_amxl)
        scot50_under, scot50_over = _under_over(y, p50_scot)
        scot70_under, scot70_over = _under_over(y, p70_scot)

        row = {
            "gl_product_group": keys[0],
            "n_asins": n_asins,
            "n_rows": n_rows,
            "total_demand": y.sum(),
            "true_mean": y.mean(),

            "p50_amxl_wape": p50_amxl_wape,
            "p50_scot_wape": p50_scot_wape,
            "p50_wape_improvement_bps": (p50_scot_wape - p50_amxl_wape) * 10000,

            "p70_amxl_wape": p70_amxl_wape,
            "p70_scot_wape": p70_scot_wape,
            "p70_wape_improvement_bps": (p70_scot_wape - p70_amxl_wape) * 10000,

            "p50_amxl_penalty": p50_amxl_pen,
            "p50_scot_penalty": p50_scot_pen,
            "p50_penalty_improvement_bps": (p50_scot_pen - p50_amxl_pen) * 10000,

            "p70_amxl_penalty": p70_amxl_pen,
            "p70_scot_penalty": p70_scot_pen,
            "p70_penalty_improvement_bps": (p70_scot_pen - p70_amxl_pen) * 10000,

            "p50_amxl_underbias": p50_under,
            "p50_amxl_overbias": p50_over,
            "p50_scot_underbias": scot50_under,
            "p50_scot_overbias": scot50_over,

            "p70_amxl_underbias": p70_under,
            "p70_amxl_overbias": p70_over,
            "p70_scot_underbias": scot70_under,
            "p70_scot_overbias": scot70_over,
        }

        if len(keys) > 1:
            row["gl_product_group_desc"] = keys[1]

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) == 0:
        print("No GL groups passed min_asins.")
        return out

    # 排序：p50 improvement 最差的在前面
    out = out.sort_values("p50_penalty_improvement_bps", ascending=True)

    display(out.round(4))
    return out
