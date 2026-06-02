# -*- coding: utf-8 -*-
"""
Chronos-2 Fine-Tuned for in_stock_dph, compared with your TCN Exposure Model.

What this script does:
1. Runs your existing TCN exposure model via run_exposure_v2(...), if requested.
2. Trains AutoGluon Chronos-2 full fine-tuning with target = in_stock_dph.
3. Merges Chronos-2 prediction with your TCN prediction.
4. Compares:
      true_instock_dph
      pred_instock_dph              # your TCN model
      chronos_pred_instock_dph      # Chronos-2 p50 prediction

Expected usage in Jupyter:
    from instock_chronos2_compare import run_full_instock_chronos2_experiment

    result = run_full_instock_chronos2_experiment(
        data_raw1=data_raw1,
        scot_df=scot_df,
        run_tcn_first=True,
        n_asins=5000,
        seed=42,
    )

    pred_df = result["tcn_pred_df"]
    chronos_pred_df = result["chronos_pred_df"]
    compare_df = result["comparison"]["compare_df"]
    summary = result["comparison"]["summary"]
    by_horizon = result["comparison"]["by_horizon"]

Notes:
- This file assumes your original TCN code is already loaded in the notebook kernel,
  including these functions:
      run_exposure_v2
      prepare_data_from_sample_scot_intersection
      filter_extreme_asins
      add_explicit_event_features
      _encode_static_features
- If you already ran your TCN model, pass run_tcn_first=False and tcn_pred_df=pred_df.
"""

import os
import gc
import shutil
import warnings
import numpy as np
import pandas as pd

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
except Exception as e:
    TimeSeriesDataFrame = None
    TimeSeriesPredictor = None
    _AUTOGLUON_IMPORT_ERROR = e
else:
    _AUTOGLUON_IMPORT_ERROR = None


# ============================================================
# 0. Metric / utility functions
# ============================================================

def _safe_numeric_chronos(s, fill=0.0):
    """Convert a pandas Series to numeric and fill missing values."""
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _wape_chronos(y, p):
    """Weighted absolute percentage error."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + 1e-8)


def _corr_chronos(y, p):
    """Pearson correlation with safe fallback."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return np.nan
    return np.corrcoef(y, p)[0, 1]


def _safe_spearman_chronos(y, p):
    """Spearman correlation implemented via ranks."""
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").values
    p_rank = pd.Series(np.asarray(p, dtype=float)).rank(method="average").values
    if np.std(y_rank) < 1e-8 or np.std(p_rank) < 1e-8:
        return np.nan
    return float(np.corrcoef(y_rank, p_rank)[0, 1])


def _auc_chronos(y_binary, score):
    """AUC for active vs zero weeks."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_binary)) < 2:
            return np.nan
        return roc_auc_score(y_binary, score)
    except Exception:
        return np.nan


def _require_autogluon():
    """Raise a clear error if AutoGluon TimeSeries is not available."""
    if TimeSeriesDataFrame is None or TimeSeriesPredictor is None:
        raise ImportError(
            "AutoGluon TimeSeries is not available in this environment. "
            "Install/import autogluon.timeseries first. Original import error: "
            f"{repr(_AUTOGLUON_IMPORT_ERROR)}"
        )


def _check_required_function(fn_name):
    """Check that a helper from your original notebook exists in global namespace."""
    if fn_name not in globals():
        raise NameError(
            f"Required function `{fn_name}` is not found. "
            "Run/import your original TCN exposure model code before using this script."
        )


# ============================================================
# 1. Data preparation for Chronos-2 target = in_stock_dph
# ============================================================

def prepare_instock_chronos2_df(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    use_glance_view_count=False,
):
    """
    Prepare dataframe and known covariates for Chronos-2 in_stock_dph prediction.

    Important:
    - target = in_stock_dph.
    - Do not include future in_stock_dph as a covariate.
    - By default, glance_view_count is NOT used to avoid future leakage.
      Set use_glance_view_count=True only if future values are known or forecasted.
    """
    _check_required_function("prepare_data_from_sample_scot_intersection")
    _check_required_function("filter_extreme_asins")
    _check_required_function("add_explicit_event_features")
    _check_required_function("_encode_static_features")

    # Same sampling and SCOT intersection as your original TCN code.
    df = prepare_data_from_sample_scot_intersection(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
    )

    # Same extreme ASIN filter as your original TCN code.
    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    df = df.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df = df.sort_values(["asin", "order_week"]).reset_index(drop=True)

    # Clean target.
    if "in_stock_dph" not in df.columns:
        raise ValueError("data_raw1 must contain column: in_stock_dph")

    df["in_stock_dph"] = _safe_numeric_chronos(df["in_stock_dph"], fill=0.0).clip(lower=0.0)
    cap = df["in_stock_dph"].quantile(dph_cap_q)
    df["in_stock_dph"] = df["in_stock_dph"].clip(upper=cap)

    # Keep true values for diagnostics / merge.
    for c in ["total_dph", "buy_box_dph", "fbi_demand"]:
        if c in df.columns:
            df[c] = _safe_numeric_chronos(df[c], fill=0.0).clip(lower=0.0)
        else:
            df[c] = 0.0

    # Demand-model-style covariates.
    if "our_price" in df.columns:
        df["our_price"] = _safe_numeric_chronos(df["our_price"], fill=0.0).clip(lower=0.0)
    else:
        df["our_price"] = 0.0

    if "scot_oos" in df.columns:
        df["scot_oos"] = _safe_numeric_chronos(df["scot_oos"], fill=0.0).clip(0, 1)
    else:
        df["scot_oos"] = 0.0

    # Calendar features.
    df["order_month"] = df["order_week"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12.0)
    df["season_winter"] = df["order_month"].isin([12, 1, 2]).astype(float)
    df["season_spring"] = df["order_month"].isin([3, 4, 5]).astype(float)
    df["season_summer"] = df["order_month"].isin([6, 7, 8]).astype(float)
    df["season_fall"] = df["order_month"].isin([9, 10, 11]).astype(float)

    # Reuse your event features.
    df, explicit_event_cols = add_explicit_event_features(
        df,
        week_col="order_week",
        event_window_weeks=4,
    )

    # Reuse your static encoding.
    df, static_cols = _encode_static_features(df)

    # Demand Chronos-2 style continuous covariates.
    chronos_demand_style_cols = [
        "historical_demand_max",
        "historical_demand_median",
        "historical_demand_min",
        "prime_demand_last",
        "prime_demand_max",
        "promotion_demand_last",
        "promotion_demand_max",
        "promotion_ratio",
        "trailing_demand_max",
        "trailing_demand_median",
        "trailing_demand_min",
    ]
    existing_demand_style_cols = [c for c in chronos_demand_style_cols if c in df.columns]

    holiday_cols = [c for c in df.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in df.columns if c.startswith("distance_")]

    optional_cols = []
    if use_glance_view_count and "glance_view_count" in df.columns:
        optional_cols.append("glance_view_count")

    known_covariates_list = list(dict.fromkeys(
        [
            "our_price",
            "scot_oos",
            "order_month",
            "month_sin",
            "month_cos",
            "season_winter",
            "season_spring",
            "season_summer",
            "season_fall",
        ]
        + explicit_event_cols
        + holiday_cols
        + distance_cols
        + static_cols
        + existing_demand_style_cols
        + optional_cols
    ))

    for c in known_covariates_list:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = _safe_numeric_chronos(df[c], fill=0.0)

    print("\n" + "=" * 100)
    print("Chronos-2 data prepared for target = in_stock_dph")
    print("=" * 100)
    print(f"ASINs: {df['asin'].nunique()}")
    print(f"Rows: {len(df)}")
    print(f"Target cap q={dph_cap_q}: {cap:.5f}")
    print(f"Known covariates count: {len(known_covariates_list)}")
    print("Known covariates:")
    for c in known_covariates_list:
        print(f"  - {c}")

    return df, known_covariates_list


# ============================================================
# 2. Build AutoGluon train / future split
# ============================================================

def build_instock_chronos2_train_future(
    df,
    known_covariates_list,
    history=52,
    horizon=20,
    target_col="in_stock_dph",
):
    """
    Build AutoGluon objects:
      - train_ts: all but the last horizon weeks for each ASIN.
      - future_known_covariates: last horizon weeks of known covariates.
      - truth_df: true target values for the last horizon weeks.
    """
    _require_autogluon()

    use_cols = ["asin", "order_week", target_col] + known_covariates_list
    work_df = df[use_cols].copy()
    work_df = work_df.rename(columns={
        "asin": "item_id",
        "order_week": "timestamp",
        target_col: "target",
    })

    work_df["item_id"] = work_df["item_id"].astype(str)
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"])
    work_df = work_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    counts = work_df.groupby("item_id").size()
    keep_asins = counts[counts >= history + horizon].index
    work_df = work_df[work_df["item_id"].isin(keep_asins)].copy()

    if work_df["item_id"].nunique() == 0:
        raise ValueError(f"No ASIN has at least history + horizon = {history + horizon} weeks.")

    train_df = (
        work_df
        .groupby("item_id", group_keys=False)
        .apply(lambda g: g.iloc[:-horizon])
        .reset_index(drop=True)
    )

    future_df = (
        work_df
        .groupby("item_id", group_keys=False)
        .apply(lambda g: g.iloc[-horizon:])
        .reset_index(drop=True)
    )

    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    future_known_covariates = TimeSeriesDataFrame.from_data_frame(
        future_df[["item_id", "timestamp"] + known_covariates_list],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    truth_df = future_df[["item_id", "timestamp", "target"]].copy()
    truth_df = truth_df.rename(columns={
        "item_id": "asin",
        "timestamp": "order_week",
        "target": "true_instock_dph_chronos_window",
    })
    truth_df["asin"] = truth_df["asin"].astype(str)
    truth_df["order_week"] = pd.to_datetime(truth_df["order_week"])
    truth_df = truth_df.sort_values(["asin", "order_week"]).reset_index(drop=True)
    truth_df["horizon"] = truth_df.groupby("asin").cumcount() + 1

    print("\n" + "=" * 100)
    print("Chronos-2 train / future split")
    print("=" * 100)
    print(f"ASINs used: {work_df['item_id'].nunique()}")
    print(f"Train rows: {len(train_df)}")
    print(f"Future rows: {len(future_df)}")
    print(f"Prediction horizon: {horizon}")

    return train_ts, future_known_covariates, truth_df


# ============================================================
# 3. Train Chronos-2 and predict in_stock_dph
# ============================================================

def train_predict_instock_chronos2(
    df,
    known_covariates_list,
    history=52,
    horizon=20,
    quantile_levels=None,
    ag_model_path="AutogluonModels/ag_InStockDPH_Chronos2_FineTuned",
    time_limit=14400,
    fine_tune_steps=2000,
    fine_tune_lr=1e-6,
):
    """
    Train one AutoGluon Chronos-2 model:
      target = in_stock_dph

    Parameters mirror your successful demand Chronos-2 configuration:
      fine_tune=True
      target_scaler='mean_abs'
      eval_during_fine_tune=True
      fine_tune_mode='full'
      fine_tune_lr=1e-6
      fine_tune_steps=2000
      eval_metric='WQL'
    """
    _require_autogluon()

    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    train_ts, future_known_covariates, truth_df = build_instock_chronos2_train_future(
        df=df,
        known_covariates_list=known_covariates_list,
        history=history,
        horizon=horizon,
        target_col="in_stock_dph",
    )

    hyperparameters = {
        "Chronos2": [
            {
                "fine_tune": True,
                "target_scaler": "mean_abs",
                "eval_during_fine_tune": True,
                "fine_tune_mode": "full",
                "fine_tune_lr": fine_tune_lr,
                "fine_tune_steps": fine_tune_steps,
                "ag_args": {"name_suffix": "FineTuned"},
            }
        ]
    }

    if os.path.exists(ag_model_path):
        print(f"Removing old model path: {ag_model_path}")
        shutil.rmtree(ag_model_path)

    predictor = TimeSeriesPredictor(
        prediction_length=horizon,
        target="target",
        known_covariates_names=known_covariates_list,
        quantile_levels=quantile_levels,
        eval_metric="WQL",
        path=ag_model_path,
    )

    print("\n" + "=" * 100)
    print("Training AutoGluon Chronos-2 for in_stock_dph")
    print("=" * 100)
    print("Hyperparameters:")
    print(hyperparameters)

    predictor.fit(
        train_data=train_ts,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        enable_ensemble=False,
    )

    print("\n" + "=" * 100)
    print("Predicting in_stock_dph with Chronos-2")
    print("=" * 100)

    pred = predictor.predict(
        train_ts,
        known_covariates=future_known_covariates,
    )

    pred_df = pred.reset_index()
    pred_df = pred_df.rename(columns={
        "item_id": "asin",
        "timestamp": "order_week",
    })

    pred_df["asin"] = pred_df["asin"].astype(str)
    pred_df["order_week"] = pd.to_datetime(pred_df["order_week"])

    if "0.5" in pred_df.columns:
        pred_df["chronos_pred_instock_dph"] = pred_df["0.5"]
    elif "mean" in pred_df.columns:
        pred_df["chronos_pred_instock_dph"] = pred_df["mean"]
    else:
        raise ValueError("AutoGluon prediction has neither column '0.5' nor 'mean'.")

    rename_q = {
        "0.1": "chronos_p10_instock_dph",
        "0.2": "chronos_p20_instock_dph",
        "0.3": "chronos_p30_instock_dph",
        "0.4": "chronos_p40_instock_dph",
        "0.5": "chronos_p50_instock_dph",
        "0.6": "chronos_p60_instock_dph",
        "0.7": "chronos_p70_instock_dph",
        "0.8": "chronos_p80_instock_dph",
        "0.9": "chronos_p90_instock_dph",
        "mean": "chronos_mean_instock_dph",
    }
    pred_df = pred_df.rename(columns={k: v for k, v in rename_q.items() if k in pred_df.columns})

    chronos_cols = [c for c in pred_df.columns if c.startswith("chronos_")]
    for c in chronos_cols:
        pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    pred_df = pred_df.sort_values(["asin", "order_week"]).reset_index(drop=True)
    pred_df["horizon"] = pred_df.groupby("asin").cumcount() + 1

    keep_cols = ["asin", "order_week", "horizon"] + chronos_cols
    chronos_pred_df = pred_df[keep_cols].copy()

    chronos_pred_df = chronos_pred_df.merge(
        truth_df,
        on=["asin", "order_week", "horizon"],
        how="left",
    )

    return predictor, chronos_pred_df


# ============================================================
# 4. Compare Chronos-2 vs your TCN
# ============================================================

def compare_instock_chronos_vs_tcn(
    tcn_pred_df,
    chronos_pred_df,
    save_path=None,
):
    """
    Merge and compare your TCN prediction with Chronos-2 prediction.

    tcn_pred_df must contain:
      asin, order_week, horizon, true_instock_dph, pred_instock_dph

    chronos_pred_df must contain:
      asin, order_week, horizon, chronos_pred_instock_dph
    """
    out = tcn_pred_df.copy()
    out["asin"] = out["asin"].astype(str)
    out["order_week"] = pd.to_datetime(out["order_week"])

    chr_df = chronos_pred_df.copy()
    chr_df["asin"] = chr_df["asin"].astype(str)
    chr_df["order_week"] = pd.to_datetime(chr_df["order_week"])

    needed_tcn_cols = [
        "asin",
        "order_week",
        "horizon",
        "true_instock_dph",
        "pred_instock_dph",
    ]
    for c in needed_tcn_cols:
        if c not in out.columns:
            raise ValueError(f"tcn_pred_df is missing column: {c}")

    if "chronos_pred_instock_dph" not in chr_df.columns:
        raise ValueError("chronos_pred_df is missing column: chronos_pred_instock_dph")

    compare_df = out.merge(
        chr_df,
        on=["asin", "order_week", "horizon"],
        how="inner",
    )

    print("\n" + "=" * 100)
    print("Merged TCN and Chronos-2 predictions")
    print("=" * 100)
    print(f"TCN rows: {len(out)}")
    print(f"Chronos rows: {len(chr_df)}")
    print(f"Merged rows: {len(compare_df)}")
    print(f"Merged ASINs: {compare_df['asin'].nunique()}")

    if len(compare_df) == 0:
        raise ValueError(
            "Merged compare_df is empty. Check whether TCN and Chronos use the same ASINs/weeks/horizon."
        )

    y = compare_df["true_instock_dph"].values.astype(float)
    p_tcn = compare_df["pred_instock_dph"].values.astype(float)
    p_chr = compare_df["chronos_pred_instock_dph"].values.astype(float)

    summary = pd.DataFrame([
        {
            "model": "Your_TCN",
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p_tcn),
            "pred_true_ratio": np.mean(p_tcn) / (np.mean(y) + 1e-8),
            "WAPE": _wape_chronos(y, p_tcn),
            "corr": _corr_chronos(y, p_tcn),
            "spearman": _safe_spearman_chronos(y, p_tcn),
            "active_AUC": _auc_chronos((y > 0).astype(int), p_tcn),
            "zero_rate_true": np.mean(y <= 0),
        },
        {
            "model": "Chronos2_FineTuned",
            "true_mean": np.mean(y),
            "pred_mean": np.mean(p_chr),
            "pred_true_ratio": np.mean(p_chr) / (np.mean(y) + 1e-8),
            "WAPE": _wape_chronos(y, p_chr),
            "corr": _corr_chronos(y, p_chr),
            "spearman": _safe_spearman_chronos(y, p_chr),
            "active_AUC": _auc_chronos((y > 0).astype(int), p_chr),
            "zero_rate_true": np.mean(y <= 0),
        },
    ])

    print("\n" + "=" * 100)
    print("IN_STOCK_DPH: Your TCN vs Chronos-2")
    print("=" * 100)
    print(summary.round(5).to_string(index=False))

    rows = []
    for h, g in compare_df.groupby("horizon"):
        y_h = g["true_instock_dph"].values.astype(float)
        tcn_h = g["pred_instock_dph"].values.astype(float)
        chr_h = g["chronos_pred_instock_dph"].values.astype(float)

        rows.append({
            "horizon": h,
            "true_mean": np.mean(y_h),
            "tcn_mean": np.mean(tcn_h),
            "tcn_ratio": np.mean(tcn_h) / (np.mean(y_h) + 1e-8),
            "tcn_WAPE": _wape_chronos(y_h, tcn_h),
            "tcn_corr": _corr_chronos(y_h, tcn_h),
            "chronos_mean": np.mean(chr_h),
            "chronos_ratio": np.mean(chr_h) / (np.mean(y_h) + 1e-8),
            "chronos_WAPE": _wape_chronos(y_h, chr_h),
            "chronos_corr": _corr_chronos(y_h, chr_h),
        })

    by_horizon = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("BY HORIZON: IN_STOCK_DPH")
    print("=" * 100)
    print(by_horizon.round(5).to_string(index=False))

    compare_df["tcn_abs_err_instock"] = np.abs(
        compare_df["true_instock_dph"] - compare_df["pred_instock_dph"]
    )
    compare_df["chronos_abs_err_instock"] = np.abs(
        compare_df["true_instock_dph"] - compare_df["chronos_pred_instock_dph"]
    )
    compare_df["chronos_minus_tcn_abs_err"] = (
        compare_df["chronos_abs_err_instock"] - compare_df["tcn_abs_err_instock"]
    )
    compare_df["chronos_better"] = (
        compare_df["chronos_abs_err_instock"] < compare_df["tcn_abs_err_instock"]
    ).astype(int)

    win_rate = compare_df["chronos_better"].mean()

    print("\n" + "=" * 100)
    print("Point-level comparison")
    print("=" * 100)
    print(f"Chronos better rate: {win_rate:.5f}")

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        compare_df.to_parquet(save_path, index=False)
        print(f"Saved compare_df to: {save_path}")

    return {
        "compare_df": compare_df,
        "summary": summary,
        "by_horizon": by_horizon,
        "chronos_better_rate": win_rate,
    }


# ============================================================
# 5. Run only Chronos-2 and compare with an existing TCN pred_df
# ============================================================

def run_instock_chronos2_compare_with_tcn(
    data_raw1,
    scot_df,
    tcn_pred_df,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    use_glance_view_count=False,
    ag_time_limit=14400,
    fine_tune_steps=2000,
    fine_tune_lr=1e-6,
    ag_model_path="AutogluonModels/ag_InStockDPH_Chronos2_FineTuned",
    save_compare_path=None,
):
    """
    Use this if you already have:
        pred_df = result_tcn["forecast_df"]
    """
    print("\n" + "=" * 100)
    print("RUN: Chronos-2 Fine-Tuned for in_stock_dph vs Your TCN")
    print("=" * 100)

    df, known_covariates_list = prepare_instock_chronos2_df(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
        dph_cap_q=dph_cap_q,
        remove_extreme=remove_extreme,
        extreme_q=extreme_q,
        use_glance_view_count=use_glance_view_count,
    )

    predictor, chronos_pred_df = train_predict_instock_chronos2(
        df=df,
        known_covariates_list=known_covariates_list,
        history=history,
        horizon=horizon,
        ag_model_path=ag_model_path,
        time_limit=ag_time_limit,
        fine_tune_steps=fine_tune_steps,
        fine_tune_lr=fine_tune_lr,
    )

    comparison = compare_instock_chronos_vs_tcn(
        tcn_pred_df=tcn_pred_df,
        chronos_pred_df=chronos_pred_df,
        save_path=save_compare_path,
    )

    gc.collect()

    return {
        "predictor": predictor,
        "chronos_pred_df": chronos_pred_df,
        "comparison": comparison,
        "known_covariates_list": known_covariates_list,
        "chronos_data": df,
    }


# ============================================================
# 6. Full experiment: run TCN first, then Chronos-2, then compare
# ============================================================

def run_full_instock_chronos2_experiment(
    data_raw1,
    scot_df,
    run_tcn_first=True,
    tcn_pred_df=None,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    # TCN params
    tcn_d_model=64,
    tcn_n_heads=4,
    tcn_batch_size=64,
    tcn_epochs=60,
    tcn_lr=1e-3,
    tcn_patience=8,
    tcn_anchor_decay=0.08,
    tcn_bce_weight=1.00,
    tcn_mag_weight=1.00,
    tcn_mean_weight=0.20,
    tcn_horizon_weight_alpha=0.40,
    tcn_high_weight_alpha=1.00,
    apply_funnel_constraint=True,
    # Shared data params
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    # Chronos params
    use_glance_view_count=False,
    ag_time_limit=14400,
    fine_tune_steps=2000,
    fine_tune_lr=1e-6,
    ag_model_path="AutogluonModels/ag_InStockDPH_Chronos2_FineTuned",
    save_compare_path="chronos2_instock_compare/compare_df.parquet",
):
    """
    One-call experiment for Jupyter.

    If run_tcn_first=True:
      - This function calls your original run_exposure_v2(...).
      - Then it trains Chronos-2 for in_stock_dph.
      - Then it compares them.

    If run_tcn_first=False:
      - You must pass tcn_pred_df=your_existing_pred_df.
    """
    if run_tcn_first:
        _check_required_function("run_exposure_v2")

        print("\n" + "=" * 100)
        print("STEP 1: Running your TCN Exposure Model")
        print("=" * 100)

        result_tcn = run_exposure_v2(
            data_raw1=data_raw1,
            scot_df=scot_df,
            n_asins=n_asins,
            seed=seed,
            history=history,
            horizon=horizon,
            d_model=tcn_d_model,
            n_heads=tcn_n_heads,
            batch_size=tcn_batch_size,
            epochs=tcn_epochs,
            lr=tcn_lr,
            patience=tcn_patience,
            dph_cap_q=dph_cap_q,
            remove_extreme=remove_extreme,
            extreme_q=extreme_q,
            apply_funnel_constraint=apply_funnel_constraint,
            anchor_decay=tcn_anchor_decay,
            bce_weight=tcn_bce_weight,
            mag_weight=tcn_mag_weight,
            mean_weight=tcn_mean_weight,
            horizon_weight_alpha=tcn_horizon_weight_alpha,
            high_weight_alpha=tcn_high_weight_alpha,
        )
        tcn_pred_df = result_tcn["forecast_df"]
    else:
        if tcn_pred_df is None:
            raise ValueError("If run_tcn_first=False, you must pass tcn_pred_df.")
        result_tcn = {"forecast_df": tcn_pred_df}

    print("\n" + "=" * 100)
    print("STEP 2: Running Chronos-2 for in_stock_dph and comparing")
    print("=" * 100)

    result_chronos = run_instock_chronos2_compare_with_tcn(
        data_raw1=data_raw1,
        scot_df=scot_df,
        tcn_pred_df=tcn_pred_df,
        n_asins=n_asins,
        seed=seed,
        history=history,
        horizon=horizon,
        dph_cap_q=dph_cap_q,
        remove_extreme=remove_extreme,
        extreme_q=extreme_q,
        use_glance_view_count=use_glance_view_count,
        ag_time_limit=ag_time_limit,
        fine_tune_steps=fine_tune_steps,
        fine_tune_lr=fine_tune_lr,
        ag_model_path=ag_model_path,
        save_compare_path=save_compare_path,
    )

    return {
        "tcn_result": result_tcn,
        "tcn_pred_df": tcn_pred_df,
        "predictor": result_chronos["predictor"],
        "chronos_pred_df": result_chronos["chronos_pred_df"],
        "comparison": result_chronos["comparison"],
        "known_covariates_list": result_chronos["known_covariates_list"],
        "chronos_data": result_chronos["chronos_data"],
    }


# ============================================================
# 7. Optional command-line guard
# ============================================================

if __name__ == "__main__":
    print(
        "This module is intended to be imported in Jupyter.\n"
        "Example:\n"
        "    from instock_chronos2_compare import run_full_instock_chronos2_experiment\n"
        "    result = run_full_instock_chronos2_experiment(data_raw1, scot_df)\n"
    )
