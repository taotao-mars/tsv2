import numpy as np
import pandas as pd

pred_df = result["forecast_df"].copy()

# 先清理可能已有的重复 GL 列，避免 merge 出 _x/_y
for c in [
    "gl_product_group",
    "gl_product_group_desc",
    "gl_product_group_x",
    "gl_product_group_y",
    "gl_product_group_desc_x",
    "gl_product_group_desc_y",
]:
    if c in pred_df.columns:
        pred_df = pred_df.drop(columns=[c])

# 只 merge 必要列
gl_cols = ["asin", "gl_product_group"]
if "gl_product_group_desc" in data_raw1.columns:
    gl_cols.append("gl_product_group_desc")

gl_map = data_raw1[gl_cols].drop_duplicates("asin").copy()

pred_df["asin"] = pred_df["asin"].astype(str)
gl_map["asin"] = gl_map["asin"].astype(str)

pred_df = pred_df.merge(gl_map, on="asin", how="left")

print("GL columns in pred_df:", [c for c in pred_df.columns if "gl" in c.lower()])

# 如果 desc 没有，就只用 gl_product_group 分组
group_cols = ["gl_product_group"]
if "gl_product_group_desc" in pred_df.columns:
    group_cols.append("gl_product_group_desc")

gl_buybox = (
    pred_df
    .groupby(group_cols, dropna=False)
    .apply(lambda g: pd.Series({
        "n_asins": g["asin"].nunique(),
        "n_rows": len(g),

        "true_mean": g["true_buy_box_dph"].mean(),
        "pred_mean": g["pred_buy_box_dph"].mean(),
        "ratio": g["pred_buy_box_dph"].sum() / (g["true_buy_box_dph"].sum() + 1e-8),

        "WAPE": (g["pred_buy_box_dph"] - g["true_buy_box_dph"]).abs().sum()
                / (g["true_buy_box_dph"].abs().sum() + 1e-8),

        "underbias": np.maximum(
            g["true_buy_box_dph"] - g["pred_buy_box_dph"], 0
        ).sum() / (g["true_buy_box_dph"].abs().sum() + 1e-8),

        "overbias": np.maximum(
            g["pred_buy_box_dph"] - g["true_buy_box_dph"], 0
        ).sum() / (g["true_buy_box_dph"].abs().sum() + 1e-8),

        "true_zero_rate": (g["true_buy_box_dph"] <= 0).mean(),
        "pred_zero_rate": (g["pred_buy_box_dph"] <= 0).mean(),

        "zero_recall": (
            ((g["true_buy_box_dph"] <= 0) & (g["pred_buy_box_dph"] <= 0)).sum()
            / max((g["true_buy_box_dph"] <= 0).sum(), 1)
        ),

        "active_recall": (
            ((g["true_buy_box_dph"] > 0) & (g["pred_buy_box_dph"] > 0)).sum()
            / max((g["true_buy_box_dph"] > 0).sum(), 1)
        ),
    }))
    .reset_index()
)

gl_buybox = gl_buybox[gl_buybox["n_asins"] >= 20].copy()

print("Most under-predicted GLs:")
display(gl_buybox.sort_values("ratio").head(10).round(4))

print("Most over-predicted GLs:")
display(gl_buybox.sort_values("ratio", ascending=False).head(10).round(4))

print("Worst WAPE GLs:")
display(gl_buybox.sort_values("WAPE", ascending=False).head(10).round(4))

print("Zero recall by GL:")
display(gl_buybox.sort_values("zero_recall").head(10).round(4))

print("Full GL buybox table:")
display(gl_buybox.sort_values("ratio").round(4))
