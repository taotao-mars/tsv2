import numpy as np
import pandas as pd
from IPython.display import display

# ============================================================
# RAW DATA GROUP-SEASONALITY EDA FOR EXPOSURE MODEL
# Focus:
#   - buy_box_dph
#   - in_stock_dph
#   - total_dph
#   - fbi_demand
#   - GL/category/month seasonality
# ============================================================

def _safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def _q(x, q):
    return np.quantile(pd.Series(x).fillna(0).values, q)


def _zero_rate(x):
    return (pd.Series(x).fillna(0) <= 0).mean()


def _active_rate(x):
    return (pd.Series(x).fillna(0) > 0).mean()


def _print_section(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def run_exposure_group_seasonality_eda(
    data_raw1,
    min_gl_asins=20,
    min_cat_asins=30,
    show_top_n=30,
    save_csv=True,
    prefix="exposure_group_seasonality_eda",
):
    df = data_raw1.copy()

    # -------------------------------
    # Basic column checks
    # -------------------------------
    required_cols = [
        "asin",
        "order_week",
        "gl_product_group",
        "category_code",
        "buy_box_dph",
        "in_stock_dph",
        "total_dph",
        "fbi_demand",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        print("Available columns:")
        print(df.columns.tolist())
        raise ValueError("Missing required columns.")

    if "gl_product_group_desc" not in df.columns:
        df["gl_product_group_desc"] = df["gl_product_group"].astype(str)

    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])

    df["order_month"] = df["order_week"].dt.month
    df["order_quarter"] = df["order_week"].dt.quarter
    df["order_year"] = df["order_week"].dt.year

    df["is_quarter_end_month"] = df["order_month"].isin([3, 6, 9, 12]).astype(int)
    df["is_holiday_q4"] = df["order_month"].isin([11, 12]).astype(int)
    df["is_summer"] = df["order_month"].isin([6, 7, 8]).astype(int)
    df["is_spring"] = df["order_month"].isin([3, 4, 5]).astype(int)
    df["is_back_to_school"] = df["order_month"].isin([8, 9]).astype(int)

    # holiday columns if present
    holiday_cols = [
        c for c in df.columns
        if c.startswith("holiday_indicator_") or c.startswith("distance_")
    ]

    _print_section("0. BASIC DATA CHECK")
    print("rows:", len(df))
    print("asins:", df["asin"].nunique())
    print("weeks:", df["order_week"].nunique())
    print("date range:", df["order_week"].min(), "→", df["order_week"].max())
    print("months:", sorted(df["order_month"].dropna().unique()))
    print("GLs:", df["gl_product_group"].nunique())
    print("categories:", df["category_code"].nunique())
    print("holiday/distance cols:", len(holiday_cols))

    print("\nMain columns summary:")
    display(
        df[["buy_box_dph", "in_stock_dph", "total_dph", "fbi_demand"]]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        .round(4)
    )

    # ============================================================
    # 1. GL × MONTH
    # ============================================================

    _print_section("1. GL × MONTH EXPOSURE / DEMAND PATTERN")

    gl_month = (
        df.groupby(["gl_product_group", "gl_product_group_desc", "order_month"], dropna=False)
        .agg(
            n_rows=("asin", "size"),
            n_asins=("asin", "nunique"),

            buybox_mean=("buy_box_dph", "mean"),
            buybox_median=("buy_box_dph", "median"),
            buybox_q90=("buy_box_dph", lambda x: _q(x, 0.90)),
            buybox_q95=("buy_box_dph", lambda x: _q(x, 0.95)),
            buybox_zero_rate=("buy_box_dph", _zero_rate),
            buybox_active_rate=("buy_box_dph", _active_rate),

            instock_mean=("in_stock_dph", "mean"),
            instock_zero_rate=("in_stock_dph", _zero_rate),

            total_mean=("total_dph", "mean"),
            total_zero_rate=("total_dph", _zero_rate),

            demand_mean=("fbi_demand", "mean"),
            demand_q90=("fbi_demand", lambda x: _q(x, 0.90)),
            demand_zero_rate=("fbi_demand", _zero_rate),
        )
        .reset_index()
    )

    gl_base = (
        df.groupby(["gl_product_group", "gl_product_group_desc"], dropna=False)
        .agg(
            gl_n_rows=("asin", "size"),
            gl_n_asins=("asin", "nunique"),

            gl_buybox_base=("buy_box_dph", "mean"),
            gl_buybox_zero_base=("buy_box_dph", _zero_rate),
            gl_buybox_q90_base=("buy_box_dph", lambda x: _q(x, 0.90)),

            gl_instock_base=("in_stock_dph", "mean"),
            gl_total_base=("total_dph", "mean"),

            gl_demand_base=("fbi_demand", "mean"),
            gl_demand_zero_base=("fbi_demand", _zero_rate),
        )
        .reset_index()
    )

    gl_month = gl_month.merge(
        gl_base,
        on=["gl_product_group", "gl_product_group_desc"],
        how="left",
    )

    gl_month["buybox_month_ratio"] = _safe_div(
        gl_month["buybox_mean"],
        gl_month["gl_buybox_base"],
    )
    gl_month["instock_month_ratio"] = _safe_div(
        gl_month["instock_mean"],
        gl_month["gl_instock_base"],
    )
    gl_month["total_month_ratio"] = _safe_div(
        gl_month["total_mean"],
        gl_month["gl_total_base"],
    )
    gl_month["demand_month_ratio"] = _safe_div(
        gl_month["demand_mean"],
        gl_month["gl_demand_base"],
    )
    gl_month["buybox_zero_lift"] = (
        gl_month["buybox_zero_rate"] - gl_month["gl_buybox_zero_base"]
    )
    gl_month["demand_zero_lift"] = (
        gl_month["demand_zero_rate"] - gl_month["gl_demand_zero_base"]
    )

    gl_month_big = gl_month[gl_month["gl_n_asins"] >= min_gl_asins].copy()

    print("GL × month table:")
    display(
        gl_month_big.sort_values(["gl_product_group", "order_month"])
        .round(4)
    )

    # ============================================================
    # 2. GL SEASONALITY VOLATILITY
    # ============================================================

    _print_section("2. GL SEASONALITY VOLATILITY")

    gl_volatility = (
        gl_month_big.groupby(["gl_product_group", "gl_product_group_desc"], dropna=False)
        .agg(
            n_months=("order_month", "nunique"),
            n_asins=("gl_n_asins", "max"),

            buybox_ratio_min=("buybox_month_ratio", "min"),
            buybox_ratio_max=("buybox_month_ratio", "max"),
            buybox_ratio_std=("buybox_month_ratio", "std"),

            instock_ratio_min=("instock_month_ratio", "min"),
            instock_ratio_max=("instock_month_ratio", "max"),
            instock_ratio_std=("instock_month_ratio", "std"),

            demand_ratio_min=("demand_month_ratio", "min"),
            demand_ratio_max=("demand_month_ratio", "max"),
            demand_ratio_std=("demand_month_ratio", "std"),

            zero_rate_min=("buybox_zero_rate", "min"),
            zero_rate_max=("buybox_zero_rate", "max"),
            zero_rate_std=("buybox_zero_rate", "std"),
        )
        .reset_index()
    )

    gl_volatility["buybox_ratio_range"] = (
        gl_volatility["buybox_ratio_max"] - gl_volatility["buybox_ratio_min"]
    )
    gl_volatility["instock_ratio_range"] = (
        gl_volatility["instock_ratio_max"] - gl_volatility["instock_ratio_min"]
    )
    gl_volatility["demand_ratio_range"] = (
        gl_volatility["demand_ratio_max"] - gl_volatility["demand_ratio_min"]
    )
    gl_volatility["zero_rate_range"] = (
        gl_volatility["zero_rate_max"] - gl_volatility["zero_rate_min"]
    )

    print("Most seasonal GLs by buybox ratio range:")
    display(
        gl_volatility.sort_values("buybox_ratio_range", ascending=False)
        .head(show_top_n)
        .round(4)
    )

    print("Most seasonal GLs by demand ratio range:")
    display(
        gl_volatility.sort_values("demand_ratio_range", ascending=False)
        .head(show_top_n)
        .round(4)
    )

    print("Most variable GLs by buybox zero-rate range:")
    display(
        gl_volatility.sort_values("zero_rate_range", ascending=False)
        .head(show_top_n)
        .round(4)
    )

    # ============================================================
    # 3. GL PEAK MONTHS
    # ============================================================

    _print_section("3. TOP BUYBOX PEAK MONTHS PER GL")

    gl_peak_month = (
        gl_month_big.sort_values("buybox_month_ratio", ascending=False)
        .groupby(["gl_product_group", "gl_product_group_desc"], dropna=False)
        .head(3)
        .sort_values(["gl_product_group", "buybox_month_ratio"], ascending=[True, False])
    )

    display(
        gl_peak_month[
            [
                "gl_product_group",
                "gl_product_group_desc",
                "order_month",
                "n_asins",
                "buybox_mean",
                "gl_buybox_base",
                "buybox_month_ratio",
                "buybox_zero_rate",
                "buybox_zero_lift",
                "demand_mean",
                "gl_demand_base",
                "demand_month_ratio",
            ]
        ].round(4)
    )

    _print_section("4. TOP BUYBOX LOW MONTHS PER GL")

    gl_low_month = (
        gl_month_big.sort_values("buybox_month_ratio", ascending=True)
        .groupby(["gl_product_group", "gl_product_group_desc"], dropna=False)
        .head(3)
        .sort_values(["gl_product_group", "buybox_month_ratio"], ascending=[True, True])
    )

    display(
        gl_low_month[
            [
                "gl_product_group",
                "gl_product_group_desc",
                "order_month",
                "n_asins",
                "buybox_mean",
                "gl_buybox_base",
                "buybox_month_ratio",
                "buybox_zero_rate",
                "buybox_zero_lift",
                "demand_month_ratio",
            ]
        ].round(4)
    )

    # ============================================================
    # 5. CATEGORY × MONTH
    # ============================================================

    _print_section("5. CATEGORY × MONTH PATTERN")

    cat_month = (
        df.groupby(["gl_product_group", "gl_product_group_desc", "category_code", "order_month"], dropna=False)
        .agg(
            n_rows=("asin", "size"),
            n_asins=("asin", "nunique"),

            buybox_mean=("buy_box_dph", "mean"),
            buybox_q90=("buy_box_dph", lambda x: _q(x, 0.90)),
            buybox_zero_rate=("buy_box_dph", _zero_rate),

            instock_mean=("in_stock_dph", "mean"),
            demand_mean=("fbi_demand", "mean"),
            demand_zero_rate=("fbi_demand", _zero_rate),
        )
        .reset_index()
    )

    cat_base = (
        df.groupby(["gl_product_group", "gl_product_group_desc", "category_code"], dropna=False)
        .agg(
            cat_n_rows=("asin", "size"),
            cat_n_asins=("asin", "nunique"),

            cat_buybox_base=("buy_box_dph", "mean"),
            cat_buybox_zero_base=("buy_box_dph", _zero_rate),
            cat_buybox_q90_base=("buy_box_dph", lambda x: _q(x, 0.90)),

            cat_demand_base=("fbi_demand", "mean"),
            cat_demand_zero_base=("fbi_demand", _zero_rate),
        )
        .reset_index()
    )

    cat_month = cat_month.merge(
        cat_base,
        on=["gl_product_group", "gl_product_group_desc", "category_code"],
        how="left",
    )

    cat_month["buybox_month_ratio"] = _safe_div(
        cat_month["buybox_mean"],
        cat_month["cat_buybox_base"],
    )
    cat_month["demand_month_ratio"] = _safe_div(
        cat_month["demand_mean"],
        cat_month["cat_demand_base"],
    )
    cat_month["buybox_zero_lift"] = (
        cat_month["buybox_zero_rate"] - cat_month["cat_buybox_zero_base"]
    )
    cat_month["demand_zero_lift"] = (
        cat_month["demand_zero_rate"] - cat_month["cat_demand_zero_base"]
    )

    cat_month_big = cat_month[cat_month["cat_n_asins"] >= min_cat_asins].copy()

    print("Most seasonal category-months by buybox month ratio:")
    display(
        cat_month_big.sort_values("buybox_month_ratio", ascending=False)
        .head(show_top_n)
        .round(4)
    )

    print("Lowest category-months by buybox month ratio:")
    display(
        cat_month_big.sort_values("buybox_month_ratio", ascending=True)
        .head(show_top_n)
        .round(4)
    )

    # ============================================================
    # 6. CATEGORY VOLATILITY
    # ============================================================

    _print_section("6. CATEGORY SEASONALITY VOLATILITY")

    cat_volatility = (
        cat_month_big.groupby(["gl_product_group", "gl_product_group_desc", "category_code"], dropna=False)
        .agg(
            n_months=("order_month", "nunique"),
            n_asins=("cat_n_asins", "max"),

            buybox_ratio_min=("buybox_month_ratio", "min"),
            buybox_ratio_max=("buybox_month_ratio", "max"),
            buybox_ratio_std=("buybox_month_ratio", "std"),

            demand_ratio_min=("demand_month_ratio", "min"),
            demand_ratio_max=("demand_month_ratio", "max"),
            demand_ratio_std=("demand_month_ratio", "std"),

            zero_rate_min=("buybox_zero_rate", "min"),
            zero_rate_max=("buybox_zero_rate", "max"),
            zero_rate_std=("buybox_zero_rate", "std"),
        )
        .reset_index()
    )

    cat_volatility["buybox_ratio_range"] = (
        cat_volatility["buybox_ratio_max"] - cat_volatility["buybox_ratio_min"]
    )
    cat_volatility["demand_ratio_range"] = (
        cat_volatility["demand_ratio_max"] - cat_volatility["demand_ratio_min"]
    )
    cat_volatility["zero_rate_range"] = (
        cat_volatility["zero_rate_max"] - cat_volatility["zero_rate_min"]
    )

    print("Most seasonal categories by buybox ratio range:")
    display(
        cat_volatility.sort_values("buybox_ratio_range", ascending=False)
        .head(show_top_n)
        .round(4)
    )

    print("Most seasonal categories by demand ratio range:")
    display(
        cat_volatility.sort_values("demand_ratio_range", ascending=False)
        .head(show_top_n)
        .round(4)
    )

    # ============================================================
    # 7. QUARTER-END MONTH CHECK
    # ============================================================

    _print_section("7. QUARTER-END MONTH CHECK BY GL")

    qend_gl = (
        df.groupby(["gl_product_group", "gl_product_group_desc", "is_quarter_end_month"], dropna=False)
        .agg(
            n_rows=("asin", "size"),
            n_asins=("asin", "nunique"),
            buybox_mean=("buy_box_dph", "mean"),
            buybox_zero_rate=("buy_box_dph", _zero_rate),
            instock_mean=("in_stock_dph", "mean"),
            demand_mean=("fbi_demand", "mean"),
        )
        .reset_index()
    )

    qend_pivot = qend_gl.pivot_table(
        index=["gl_product_group", "gl_product_group_desc"],
        columns="is_quarter_end_month",
        values=["n_asins", "buybox_mean", "buybox_zero_rate", "instock_mean", "demand_mean"],
        aggfunc="first",
    )

    qend_pivot.columns = [
        f"{a}_{b}" for a, b in qend_pivot.columns
    ]
    qend_pivot = qend_pivot.reset_index()

    if "buybox_mean_1" in qend_pivot.columns and "buybox_mean_0" in qend_pivot.columns:
        qend_pivot["qend_buybox_ratio"] = _safe_div(
            qend_pivot["buybox_mean_1"],
            qend_pivot["buybox_mean_0"],
        )
    if "demand_mean_1" in qend_pivot.columns and "demand_mean_0" in qend_pivot.columns:
        qend_pivot["qend_demand_ratio"] = _safe_div(
            qend_pivot["demand_mean_1"],
            qend_pivot["demand_mean_0"],
        )

    display(
        qend_pivot.sort_values("qend_buybox_ratio", ascending=False)
        .round(4)
    )

    # ============================================================
    # 8. HOLIDAY / SPECIAL MONTH CHECK
    # ============================================================

    _print_section("8. SPECIAL MONTH CHECK BY GL")

    special_flags = [
        "is_holiday_q4",
        "is_summer",
        "is_spring",
        "is_back_to_school",
    ]

    special_tables = {}

    for flag in special_flags:
        tmp = (
            df.groupby(["gl_product_group", "gl_product_group_desc", flag], dropna=False)
            .agg(
                n_rows=("asin", "size"),
                n_asins=("asin", "nunique"),
                buybox_mean=("buy_box_dph", "mean"),
                buybox_zero_rate=("buy_box_dph", _zero_rate),
                demand_mean=("fbi_demand", "mean"),
            )
            .reset_index()
        )

        piv = tmp.pivot_table(
            index=["gl_product_group", "gl_product_group_desc"],
            columns=flag,
            values=["n_asins", "buybox_mean", "buybox_zero_rate", "demand_mean"],
            aggfunc="first",
        )

        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
        piv = piv.reset_index()

        if "buybox_mean_1" in piv.columns and "buybox_mean_0" in piv.columns:
            piv[f"{flag}_buybox_ratio"] = _safe_div(
                piv["buybox_mean_1"],
                piv["buybox_mean_0"],
            )
        if "demand_mean_1" in piv.columns and "demand_mean_0" in piv.columns:
            piv[f"{flag}_demand_ratio"] = _safe_div(
                piv["demand_mean_1"],
                piv["demand_mean_0"],
            )

        special_tables[flag] = piv

        print(f"\nSpecial flag: {flag}")
        display(
            piv.sort_values(f"{flag}_buybox_ratio", ascending=False)
            .head(show_top_n)
            .round(4)
        )

    # ============================================================
    # 9. MONTH PROFILE MATRIX
    # ============================================================

    _print_section("9. GL MONTH RATIO MATRIX")

    gl_month_matrix = gl_month_big.pivot_table(
        index=["gl_product_group", "gl_product_group_desc"],
        columns="order_month",
        values="buybox_month_ratio",
        aggfunc="mean",
    ).reset_index()

    display(gl_month_matrix.round(3))

    _print_section("10. CATEGORY MONTH RATIO MATRIX")

    cat_month_matrix = cat_month_big.pivot_table(
        index=["gl_product_group", "gl_product_group_desc", "category_code"],
        columns="order_month",
        values="buybox_month_ratio",
        aggfunc="mean",
    ).reset_index()

    display(cat_month_matrix.round(3).head(100))

    # ============================================================
    # 11. FEATURE USEFULNESS SUMMARY
    # ============================================================

    _print_section("11. FEATURE USEFULNESS SUMMARY")

    gl_summary = pd.DataFrame([{
        "n_gl_big": gl_month_big[["gl_product_group", "gl_product_group_desc"]].drop_duplicates().shape[0],
        "median_gl_buybox_ratio_range": gl_volatility["buybox_ratio_range"].median(),
        "max_gl_buybox_ratio_range": gl_volatility["buybox_ratio_range"].max(),
        "median_gl_zero_rate_range": gl_volatility["zero_rate_range"].median(),
        "max_gl_zero_rate_range": gl_volatility["zero_rate_range"].max(),
        "median_gl_demand_ratio_range": gl_volatility["demand_ratio_range"].median(),
        "max_gl_demand_ratio_range": gl_volatility["demand_ratio_range"].max(),
    }])

    cat_summary = pd.DataFrame([{
        "n_cat_big": cat_month_big[["gl_product_group", "category_code"]].drop_duplicates().shape[0],
        "median_cat_buybox_ratio_range": cat_volatility["buybox_ratio_range"].median(),
        "max_cat_buybox_ratio_range": cat_volatility["buybox_ratio_range"].max(),
        "median_cat_zero_rate_range": cat_volatility["zero_rate_range"].median(),
        "max_cat_zero_rate_range": cat_volatility["zero_rate_range"].max(),
        "median_cat_demand_ratio_range": cat_volatility["demand_ratio_range"].median(),
        "max_cat_demand_ratio_range": cat_volatility["demand_ratio_range"].max(),
    }])

    print("GL seasonality summary:")
    display(gl_summary.round(4))

    print("Category seasonality summary:")
    display(cat_summary.round(4))

    print("\nInterpretation guide:")
    print("- buybox_month_ratio > 1 means this group/month has higher buybox exposure than its group average.")
    print("- buybox_month_ratio < 1 means this group/month is below its group average.")
    print("- buybox_ratio_range large means strong group-specific seasonality.")
    print("- zero_rate_range large means zero/active pattern changes by month.")
    print("- category range much larger than GL range means category_code is useful beyond GL.")

    # ============================================================
    # 12. SAVE
    # ============================================================

    outputs = {
        "gl_month": gl_month,
        "gl_month_big": gl_month_big,
        "gl_volatility": gl_volatility,
        "gl_peak_month": gl_peak_month,
        "gl_low_month": gl_low_month,
        "cat_month": cat_month,
        "cat_month_big": cat_month_big,
        "cat_volatility": cat_volatility,
        "qend_pivot": qend_pivot,
        "special_tables": special_tables,
        "gl_month_matrix": gl_month_matrix,
        "cat_month_matrix": cat_month_matrix,
        "gl_summary": gl_summary,
        "cat_summary": cat_summary,
    }

    if save_csv:
        gl_month.to_csv(f"{prefix}_gl_month.csv", index=False)
        gl_volatility.to_csv(f"{prefix}_gl_volatility.csv", index=False)
        gl_peak_month.to_csv(f"{prefix}_gl_peak_month.csv", index=False)
        gl_low_month.to_csv(f"{prefix}_gl_low_month.csv", index=False)

        cat_month.to_csv(f"{prefix}_cat_month.csv", index=False)
        cat_volatility.to_csv(f"{prefix}_cat_volatility.csv", index=False)

        qend_pivot.to_csv(f"{prefix}_qend_pivot.csv", index=False)
        gl_month_matrix.to_csv(f"{prefix}_gl_month_matrix.csv", index=False)
        cat_month_matrix.to_csv(f"{prefix}_cat_month_matrix.csv", index=False)

        for flag, table in special_tables.items():
            table.to_csv(f"{prefix}_{flag}.csv", index=False)

        print("\nSaved CSV files with prefix:", prefix)

    return outputs


# ============================================================
# RUN
# ============================================================

season_eda = run_exposure_group_seasonality_eda(
    data_raw1=data_raw1,
    min_gl_asins=20,
    min_cat_asins=30,
    show_top_n=30,
    save_csv=True,
    prefix="exposure_group_seasonality_eda",
)
