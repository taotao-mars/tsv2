import numpy as np
import pandas as pd

pred_df = result["forecast_df"].copy()

# 如果 pred_df 里还没有 gl_product_group，就从 data_raw1 合并进来
if "gl_product_group" not in pred_df.columns:
    gl_map = data_raw1[["asin", "gl_product_group", "gl_product_group_desc"]].drop_duplicates("asin")
    pred_df["asin"] = pred_df["asin"].astype(str)
    gl_map["asin"] = gl_map["asin"].astype(str)
    pred_df = pred_df.merge(gl_map, on="asin", how="left")

# 看 buy_box exposure 在不同 GL 下表现
gl_buybox = (
    pred_df
    .groupby(["gl_product_group", "gl_product_group_desc"], dropna=False)
    .apply(lambda g: pd.Series({
        "n_asins": g["asin"].nunique(),
        "n_rows": len(g),

        "true_mean": g["true_buy_box_dph"].mean(),
        "pred_mean": g["pred_buy_box_dph"].mean(),
        "ratio": g["pred_buy_box_dph"].sum() / (g["true_buy_box_dph"].sum() + 1e-8),

        "WAPE": (g["pred_buy_box_dph"] - g["true_buy_box_dph"]).abs().sum()
                / (g["true_buy_box_dph"].abs().sum() + 1e-8),

        "underbias": np.maximum(g["true_buy_box_dph"] - g["pred_buy_box_dph"], 0).sum()
                     / (g["true_buy_box_dph"].abs().sum() + 1e-8),

        "overbias": np.maximum(g["pred_buy_box_dph"] - g["true_buy_box_dph"], 0).sum()
                    / (g["true_buy_box_dph"].abs().sum() + 1e-8),

        "true_zero_rate": (g["true_buy_box_dph"] <= 0).mean(),
        "pred_zero_rate": (g["pred_buy_box_dph"] <= 0).mean(),

        "zero_recall": ((g["true_buy_box_dph"] <= 0) & (g["pred_buy_box_dph"] <= 0)).sum()
                       / max((g["true_buy_box_dph"] <= 0).sum(), 1),

        "active_recall": ((g["true_buy_box_dph"] > 0) & (g["pred_buy_box_dph"] > 0)).sum()
                         / max((g["true_buy_box_dph"] > 0).sum(), 1),
    }))
    .reset_index()
)

# 只看样本足够的 GL
gl_buybox = gl_buybox[gl_buybox["n_asins"] >= 20].copy()

print("Most under-predicted GLs:")
display(gl_buybox.sort_values("ratio").head(10).round(4))

print("Most over-predicted GLs:")
display(gl_buybox.sort_values("ratio", ascending=False).head(10).round(4))

print("Worst WAPE GLs:")
display(gl_buybox.sort_values("WAPE", ascending=False).head(10).round(4))

print("Zero recall by GL:")
display(gl_buybox.sort_values("zero_recall").head(10).round(4))
