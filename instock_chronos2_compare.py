import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import shutil
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def _safe_numeric_chronos(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _wape_chronos(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + 1e-8)


def _corr_chronos(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def _safe_spearman_chronos(y, p):
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").values
    p_rank = pd.Series(np.asarray(p, dtype=float)).rank(method="average").values
    if np.std(y_rank) < 1e-8 or np.std(p_rank) < 1e-8:
        return np.nan
    return float(np.corrcoef(y_rank, p_rank)[0, 1])


def _auc_chronos(y_binary, score):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_binary)) < 2:
            return np.nan
        return float(roc_auc_score(y_binary, score))
    except Exception:
        return np.nan


def _check_required_old_functions():
    required = [
        "prepare_data_from_sample_scot_intersection",
        "filter_extreme_asins",
        "add_explicit_event_features",
        "_encode_static_features",
        "run_exposure_v2",
    ]
    missing = [x for x in required if x not in globals()]
    if missing:
        raise NameError("Missing functions:\n" + "\n".join(missing))


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
    _check_required_old_functions()

    df = prepare_data_from_sample_scot_intersection(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
    )

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    df = df.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df = df.sort_values(["asin", "order_week"]).reset_index(drop=True)

    if "in_stock_dph" not in df.columns:
        raise ValueError("data_raw1 must contain in_stock_dph")

    df["in_stock_dph"] = _safe_numeric_chronos(df["in_stock_dph"]).clip(lower=0.0)
    cap = df["in_stock_dph"].quantile(dph_cap_q)
    df["in_stock_dph"] = df["in_stock_dph"].clip(upper=cap)

    for c in ["total_dph", "buy_box_dph", "fbi_demand"]:
        if c in df.columns:
            df[c] = _safe_numeric_chronos(df[c]).clip(lower=0.0)
        else:
            df[c] = 0.0

    df["our_price"] = _safe_numeric_chronos(df["our_price"] if "our_price" in df.columns else 0.0).clip(lower=0.0)
    df["scot_oos"] = _safe_numeric_chronos(df["scot_oos"] if "scot_oos" in df.columns else 0.0).clip(0, 1)

    df["order_month"] = df["order_week"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12.0)
    df["season_winter"] = df["order_month"].isin([12, 1, 2]).astype(float)
    df["season_spring"] = df["order_month"].isin([3, 4, 5]).astype(float)
    df["season_summer"] = df["order_month"].isin([6, 7, 8]).astype(float)
    df["season_fall"] = df["order_month"].isin([9, 10, 11]).astype(float)

    df, explicit_event_cols = add_explicit_event_features(
        df,
        week_col="order_week",
        event_window_weeks=4,
    )

    df, static_cols = _encode_static_features(df)

    demand_style_cols = [
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

    existing_demand_style_cols = [c for c in demand_style_cols if c in df.columns]
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
        df[c] = _safe_numeric_chronos(df[c])

    print("=" * 100)
    print("Chronos data ready")
    print(f"ASINs: {df['asin'].nunique()} | Rows: {len(df)} | Covariates: {len(known_covariates_list)}")
    print("Target cap:", cap)
    print("=" * 100)

    return df, known_covariates_list


def build_instock_chronos2_train_future(
    df,
    known_covariates_list,
    history=52,
    horizon=20,
    target_col="in_stock_dph",
):
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
        raise ValueError(f"No ASIN has at least {history + horizon} weeks")

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

    print("=" * 100)
    print("Chronos split ready")
    print(f"ASINs used: {work_df['item_id'].nunique()}")
    print(f"Train rows: {len(train_df)} | Future rows: {len(future_df)}")
    print("=" * 100)

    return train_ts, future_known_covariates, truth_df


def train_predict_instock_chronos2(
    df,
    known_covariates_list,
    history=52,
    horizon=20,
    quantile_levels=None,
    main_quantile="0.5",
    ag_model_path="AutogluonModels/ag_InStockDPH_Chronos2_FineTuned",
    time_limit=14400,
    fine_tune_steps=2000,
    fine_tune_lr=1e-6,
):
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
        shutil.rmtree(ag_model_path)

    predictor = TimeSeriesPredictor(
        prediction_length=horizon,
        target="target",
        known_covariates_names=known_covariates_list,
        quantile_levels=quantile_levels,
        eval_metric="WQL",
        path=ag_model_path,
    )

    predictor.fit(
        train_data=train_ts,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        enable_ensemble=False,
    )

    try:
        lb = predictor.leaderboard(silent=True)
        print("=" * 100)
        print("Chronos leaderboard")
        print(lb)
        print("=" * 100)

        if lb is None or len(lb) == 0:
            raise RuntimeError("Chronos training failed: empty leaderboard")

    except Exception as e:
        raise RuntimeError("Chronos training failed before prediction") from e

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

    col_map = {str(c): c for c in pred_df.columns}

    if main_quantile in col_map:
        pred_df["chronos_pred_instock_dph"] = pred_df[col_map[main_quantile]]
    elif "0.5" in col_map:
        pred_df["chronos_pred_instock_dph"] = pred_df[col_map["0.5"]]
    elif "mean" in pred_df.columns:
        pred_df["chronos_pred_instock_dph"] = pred_df["mean"]
    else:
        raise ValueError("No main quantile / p50 / mean found")

    q_rename = {
        "0.1": "chronos_p10_instock_dph",
        "0.2": "chronos_p20_instock_dph",
        "0.3": "chronos_p30_instock_dph",
        "0.4": "chronos_p40_instock_dph",
        "0.5": "chronos_p50_instock_dph",
        "0.6": "chronos_p60_instock_dph",
        "0.7": "chronos_p70_instock_dph",
        "0.8": "chronos_p80_instock_dph",
        "0.9": "chronos_p90_instock_dph",
    }

    for q, new_col in q_rename.items():
        if q in col_map:
            pred_df[new_col] = pred_df[col_map[q]]

    if "mean" in pred_df.columns:
        pred_df["chronos_mean_instock_dph"] = pred_df["mean"]

    chronos_cols = [c for c in pred_df.columns if c.startswith("chronos_")]

    for c in chronos_cols:
        pred_df[c] = (
            pd.to_numeric(pred_df[c], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

    pred_df = pred_df.sort_values(["asin", "order_week"]).reset_index(drop=True)
    pred_df["horizon"] = pred_df.groupby("asin").cumcount() + 1

    chronos_pred_df = pred_df[["asin", "order_week", "horizon"] + chronos_cols].copy()

    chronos_pred_df = chronos_pred_df.merge(
        truth_df,
        on=["asin", "order_week", "horizon"],
        how="left",
    )

    return predictor, chronos_pred_df


def compare_instock_chronos_vs_tcn(
    tcn_pred_df,
    chronos_pred_df,
    save_path=None,
):
    out = tcn_pred_df.copy()
    out["asin"] = out["asin"].astype(str)
    out["order_week"] = pd.to_datetime(out["order_week"])

    chr_df = chronos_pred_df.copy()
    chr_df["asin"] = chr_df["asin"].astype(str)
    chr_df["order_week"] = pd.to_datetime(chr_df["order_week"])

    required = ["asin", "order_week", "horizon", "true_instock_dph", "pred_instock_dph"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError("tcn_pred_df missing columns: " + str(missing))

    compare_df = out.merge(
        chr_df,
        on=["asin", "order_week", "horizon"],
        how="inner",
    )

    if len(compare_df) == 0:
        raise ValueError("compare_df is empty after merge")

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

    compare_df["tcn_abs_err_instock"] = np.abs(
        compare_df["true_instock_dph"] - compare_df["pred_instock_dph"]
    )

    compare_df["chronos_abs_err_instock"] = np.abs(
        compare_df["true_instock_dph"] - compare_df["chronos_pred_instock_dph"]
    )

    compare_df["chronos_minus_tcn_abs_err"] = (
        compare_df["chronos_abs_err_instock"]
        - compare_df["tcn_abs_err_instock"]
    )

    compare_df["chronos_better"] = (
        compare_df["chronos_abs_err_instock"]
        < compare_df["tcn_abs_err_instock"]
    ).astype(int)

    win_rate = compare_df["chronos_better"].mean()

    print("=" * 100)
    print("Merged rows:", len(compare_df))
    print("Merged ASINs:", compare_df["asin"].nunique())
    print("Chronos better rate:", round(win_rate, 5))
    print("=" * 100)
    print(summary.round(5).to_string(index=False))
    print("=" * 100)
    print(by_horizon.round(5).to_string(index=False))

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if save_path.endswith(".csv"):
            compare_df.to_csv(save_path, index=False)
        else:
            compare_df.to_parquet(save_path, index=False)

        print("Saved:", save_path)

    return {
        "compare_df": compare_df,
        "summary": summary,
        "by_horizon": by_horizon,
        "chronos_better_rate": win_rate,
    }


N_ASINS = 5000
SEED = 42
HISTORY = 52
HORIZON = 20

AG_TIME_LIMIT = 14400
FINE_TUNE_STEPS = 2000
FINE_TUNE_LR = 1e-6

CHRONOS_MAIN_QUANTILE = "0.5"   # "0.5" or "0.7"
USE_GLANCE_VIEW_COUNT = False

AG_MODEL_PATH = "AutogluonModels/ag_InStockDPH_Chronos2_FineTuned"
SAVE_COMPARE_PATH = "chronos2_instock_compare_df.parquet"


chronos_df, known_covariates_list = prepare_instock_chronos2_df(
    data_raw1=data_raw1,
    scot_df=scot_df,
    n_asins=N_ASINS,
    seed=SEED,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    use_glance_view_count=USE_GLANCE_VIEW_COUNT,
)

chronos_predictor, chronos_pred_df = train_predict_instock_chronos2(
    df=chronos_df,
    known_covariates_list=known_covariates_list,
    history=HISTORY,
    horizon=HORIZON,
    main_quantile=CHRONOS_MAIN_QUANTILE,
    ag_model_path=AG_MODEL_PATH,
    time_limit=AG_TIME_LIMIT,
    fine_tune_steps=FINE_TUNE_STEPS,
    fine_tune_lr=FINE_TUNE_LR,
)

print("Chronos done:", chronos_pred_df.shape)
display(chronos_pred_df.head())

result_tcn = run_exposure_v2(
    data_raw1=data_raw1,
    scot_df=scot_df,
    n_asins=N_ASINS,
    seed=SEED,
    history=HISTORY,
    horizon=HORIZON,
    d_model=64,
    n_heads=4,
    batch_size=64,
    epochs=60,
    lr=1e-3,
    patience=8,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    apply_funnel_constraint=True,
    anchor_decay=0.08,
    bce_weight=1.00,
    mag_weight=1.00,
    mean_weight=0.20,
    horizon_weight_alpha=0.40,
    high_weight_alpha=1.00,
)

pred_df = result_tcn["forecast_df"]

comparison = compare_instock_chronos_vs_tcn(
    tcn_pred_df=pred_df,
    chronos_pred_df=chronos_pred_df,
    save_path=SAVE_COMPARE_PATH,
)

compare_df = comparison["compare_df"]
summary = comparison["summary"]
by_horizon = comparison["by_horizon"]

result_chronos_instock = {
    "predictor": chronos_predictor,
    "chronos_pred_df": chronos_pred_df,
    "comparison": comparison,
    "known_covariates_list": known_covariates_list,
    "chronos_data": chronos_df,
}

display(summary.round(5))
display(by_horizon.round(5))

display(
    compare_df[
        [
            "asin",
            "order_week",
            "horizon",
            "true_instock_dph",
            "pred_instock_dph",
            "chronos_pred_instock_dph",
            "tcn_abs_err_instock",
            "chronos_abs_err_instock",
            "chronos_better",
        ]
    ].head(50)
)
