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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


torch.manual_seed(42)
np.random.seed(42)


def _safe_numeric(s, fill=0.0):
    if isinstance(s, (int, float)):
        return s
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _wape(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.sum(np.abs(y - p)) / (np.sum(np.abs(y)) + 1e-8)


def _corr(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def _safe_spearman(y, p):
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").values
    p_rank = pd.Series(np.asarray(p, dtype=float)).rank(method="average").values
    if np.std(y_rank) < 1e-8 or np.std(p_rank) < 1e-8:
        return np.nan
    return float(np.corrcoef(y_rank, p_rank)[0, 1])


def _auc(y_binary, score):
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_binary)) < 2:
            return np.nan
        return float(roc_auc_score(y_binary, score))
    except Exception:
        return np.nan


def prepare_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins=5000, seed=42):
    df = data_raw1.copy()
    scot = scot_df.copy()
    df["asin"] = df["asin"].astype(str)
    scot["asin"] = scot["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])

    rng = np.random.default_rng(seed)
    unique_asins = df["asin"].dropna().unique()
    sample_asins = rng.choice(unique_asins, size=min(n_asins, len(unique_asins)), replace=False)

    sample_asin_set = set(sample_asins)
    scot_asin_set = set(scot["asin"].dropna().unique())
    intersect_asins = sorted(sample_asin_set & scot_asin_set)

    print(f"Sample ASINs: {len(sample_asin_set)} | SCOT ASINs: {len(scot_asin_set)} | Intersection: {len(intersect_asins)}")
    return df[df["asin"].isin(intersect_asins)].copy()


def filter_extreme_asins(data_raw, q=0.99):
    df = data_raw.copy()

    for c in ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    stats = (
        df.groupby("asin")
        .agg(
            max_demand=("fbi_demand", "max"),
            max_total_dph=("total_dph", "max"),
            max_buy_box_dph=("buy_box_dph", "max"),
            max_instock_dph=("in_stock_dph", "max"),
        )
        .reset_index()
    )

    thresholds = {
        c: stats[c].quantile(q)
        for c in ["max_demand", "max_total_dph", "max_buy_box_dph", "max_instock_dph"]
    }

    keep = stats[
        (stats["max_demand"] <= thresholds["max_demand"])
        & (stats["max_total_dph"] <= thresholds["max_total_dph"])
        & (stats["max_buy_box_dph"] <= thresholds["max_buy_box_dph"])
        & (stats["max_instock_dph"] <= thresholds["max_instock_dph"])
    ]["asin"]

    out = df[df["asin"].isin(set(keep))].copy()
    print(f"Extreme filter: {df['asin'].nunique()} -> {out['asin'].nunique()} ASINs")
    return out


def _encode_static_features(df):
    df = df.copy()
    out_cols = []

    for c in ["gl_product_group", "ind_top10_brand"]:
        if c not in df.columns:
            continue

        raw = df[c].astype(str).fillna("MISSING")
        codes, uniques = pd.factorize(raw)
        denom = max(len(uniques) - 1, 1)

        code_col = f"stock_static__{c}__code"
        freq_col = f"stock_static__{c}__freq"

        df[code_col] = codes.astype(float) / denom
        freq = raw.value_counts(normalize=True)
        df[freq_col] = raw.map(freq).fillna(0.0).astype(float)

        out_cols.extend([code_col, freq_col])

    return df, out_cols


def _event_thanksgiving_date(year):
    nov = pd.date_range(f"{year}-11-01", f"{year}-11-30", freq="D")
    return nov[nov.weekday == 3][3]


def _make_event_calendar(min_year, max_year):
    events = []
    for y in range(min_year - 1, max_year + 2):
        tg = _event_thanksgiving_date(y)
        events += [
            ("event_NewYear", pd.Timestamp(f"{y}-01-01")),
            ("event_PrimeDay_proxy_July", pd.Timestamp(f"{y}-07-15")),
            ("event_BackToSchool_proxy", pd.Timestamp(f"{y}-08-15")),
            ("event_Thanksgiving", tg),
            ("event_BlackFriday", tg + pd.Timedelta(days=1)),
            ("event_CyberMonday", tg + pd.Timedelta(days=4)),
            ("event_Christmas", pd.Timestamp(f"{y}-12-25")),
        ]

    ev = pd.DataFrame(events, columns=["event_name", "event_date"])
    ev["event_week"] = ev["event_date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)
    return ev


def add_explicit_event_features(df, week_col="order_week", event_window_weeks=4):
    out = df.copy()
    out[week_col] = pd.to_datetime(out[week_col])
    out["week_start"] = out[week_col].dt.to_period("W-SUN").apply(lambda r: r.start_time)

    events = _make_event_calendar(out[week_col].dt.year.min(), out[week_col].dt.year.max())
    event_names = sorted(events["event_name"].unique().tolist())

    out["is_event_window"] = 0.0
    out["weeks_to_nearest_event"] = 99.0
    out["abs_weeks_to_nearest_event"] = 99.0
    out["is_pre_event"] = 0.0
    out["is_post_event"] = 0.0
    out["pre_event_proximity"] = 0.0
    out["post_event_decay"] = 0.0

    for ev_name in event_names:
        out[f"{ev_name}_window"] = 0.0
        out[f"{ev_name}_week_exact"] = 0.0

    for _, r in events.iterrows():
        ev_name = r["event_name"]
        ev_week = r["event_week"]
        diff = ((out["week_start"] - ev_week).dt.days / 7).round().astype(int)

        in_window = diff.abs() <= event_window_weeks
        exact_week = diff == 0

        out.loc[in_window, "is_event_window"] = 1.0
        out.loc[in_window, f"{ev_name}_window"] = 1.0
        out.loc[exact_week, f"{ev_name}_week_exact"] = 1.0

        current_abs = out["abs_weeks_to_nearest_event"].astype(float)
        new_abs = diff.abs().astype(float)
        replace = new_abs < current_abs

        out.loc[replace, "weeks_to_nearest_event"] = diff[replace].astype(float)
        out.loc[replace, "abs_weeks_to_nearest_event"] = new_abs[replace].astype(float)

    out["is_pre_event"] = ((out["weeks_to_nearest_event"] < 0) & (out["is_event_window"] > 0)).astype(float)
    out["is_post_event"] = ((out["weeks_to_nearest_event"] > 0) & (out["is_event_window"] > 0)).astype(float)

    weeks_raw = out["weeks_to_nearest_event"].astype(float)
    out["pre_event_proximity"] = np.exp(-0.15 * (-weeks_raw).clip(lower=0.0))
    out["post_event_decay"] = np.exp(-0.15 * weeks_raw.clip(lower=0.0))

    out["weeks_to_nearest_event"] = out["weeks_to_nearest_event"].clip(-20, 20) / 20.0
    out["abs_weeks_to_nearest_event"] = out["abs_weeks_to_nearest_event"].clip(0, 20) / 20.0

    event_cols = (
        [
            "is_event_window",
            "weeks_to_nearest_event",
            "abs_weeks_to_nearest_event",
            "is_pre_event",
            "is_post_event",
            "pre_event_proximity",
            "post_event_decay",
        ]
        + [f"{ev_name}_window" for ev_name in event_names]
        + [f"{ev_name}_week_exact" for ev_name in event_names]
    )

    return out, event_cols


def prepare_chronos_df(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    dph_cap_q=0.995,
    remove_extreme=True,
    extreme_q=0.99,
    use_glance_view_count=False,
):
    df = prepare_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    df = df.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df = df.sort_values(["asin", "order_week"]).reset_index(drop=True)

    for c in ["in_stock_dph", "total_dph", "buy_box_dph", "fbi_demand"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    cap = df["in_stock_dph"].quantile(dph_cap_q)
    df["in_stock_dph"] = df["in_stock_dph"].clip(upper=cap)

    if "our_price" in df.columns:
        df["our_price"] = pd.to_numeric(df["our_price"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["our_price"] = 0.0

    if "scot_oos" in df.columns:
        df["scot_oos"] = pd.to_numeric(df["scot_oos"], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        df["scot_oos"] = 0.0

    df["order_month"] = df["order_week"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12.0)
    df["season_winter"] = df["order_month"].isin([12, 1, 2]).astype(float)
    df["season_spring"] = df["order_month"].isin([3, 4, 5]).astype(float)
    df["season_summer"] = df["order_month"].isin([6, 7, 8]).astype(float)
    df["season_fall"] = df["order_month"].isin([9, 10, 11]).astype(float)

    df, event_cols = add_explicit_event_features(df, week_col="order_week", event_window_weeks=4)
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

    known_covariates = list(dict.fromkeys(
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
        + event_cols
        + holiday_cols
        + distance_cols
        + static_cols
        + existing_demand_style_cols
        + optional_cols
    ))

    for c in known_covariates:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    print("=" * 100)
    print("Chronos data ready")
    print(f"ASINs: {df['asin'].nunique()} | Rows: {len(df)} | Covariates: {len(known_covariates)} | Target cap: {cap:.4f}")
    print("=" * 100)

    return df, known_covariates


def build_chronos_split(df, known_covariates, history=52, horizon=20):
    work_df = df[["asin", "order_week", "in_stock_dph"] + known_covariates].copy()
    work_df = work_df.rename(columns={"asin": "item_id", "order_week": "timestamp", "in_stock_dph": "target"})

    work_df["item_id"] = work_df["item_id"].astype(str)
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"])
    work_df = work_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    counts = work_df.groupby("item_id").size()
    keep = counts[counts >= history + horizon].index
    work_df = work_df[work_df["item_id"].isin(keep)].copy()

    if work_df["item_id"].nunique() == 0:
        raise ValueError(f"No ASIN has at least {history + horizon} weeks")

    train_df = work_df.groupby("item_id", group_keys=False).apply(lambda g: g.iloc[:-horizon]).reset_index(drop=True)
    future_df = work_df.groupby("item_id", group_keys=False).apply(lambda g: g.iloc[-horizon:]).reset_index(drop=True)

    train_ts = TimeSeriesDataFrame.from_data_frame(train_df, id_column="item_id", timestamp_column="timestamp")
    future_cov = TimeSeriesDataFrame.from_data_frame(
        future_df[["item_id", "timestamp"] + known_covariates],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    truth_df = future_df[["item_id", "timestamp", "target"]].copy()
    truth_df = truth_df.rename(columns={"item_id": "asin", "timestamp": "order_week", "target": "true_instock_dph_chronos_window"})
    truth_df["asin"] = truth_df["asin"].astype(str)
    truth_df["order_week"] = pd.to_datetime(truth_df["order_week"])
    truth_df = truth_df.sort_values(["asin", "order_week"]).reset_index(drop=True)
    truth_df["horizon"] = truth_df.groupby("asin").cumcount() + 1

    print("=" * 100)
    print("Chronos split ready")
    print(f"ASINs used: {work_df['item_id'].nunique()} | Train rows: {len(train_df)} | Future rows: {len(future_df)}")
    print("=" * 100)

    return train_ts, future_cov, truth_df


def train_predict_chronos(
    df,
    known_covariates,
    history=52,
    horizon=20,
    main_quantile="0.5",
    ag_model_path="AutogluonModels/ag_InStockDPH_Chronos2_FineTuned",
    time_limit=14400,
    fine_tune_steps=2000,
    fine_tune_lr=1e-6,
):
    train_ts, future_cov, truth_df = build_chronos_split(df, known_covariates, history, horizon)

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
        known_covariates_names=known_covariates,
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        eval_metric="WQL",
        path=ag_model_path,
    )

    predictor.fit(
        train_data=train_ts,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        enable_ensemble=False,
    )

    lb = predictor.leaderboard(silent=True)
    print("=" * 100)
    print("Chronos leaderboard")
    print(lb)
    print("=" * 100)

    if lb is None or len(lb) == 0:
        raise RuntimeError("Chronos failed: empty leaderboard")

    pred = predictor.predict(train_ts, known_covariates=future_cov)

    pred_df = pred.reset_index()
    pred_df = pred_df.rename(columns={"item_id": "asin", "timestamp": "order_week"})
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
        pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    pred_df = pred_df.sort_values(["asin", "order_week"]).reset_index(drop=True)
    pred_df["horizon"] = pred_df.groupby("asin").cumcount() + 1

    chronos_pred_df = pred_df[["asin", "order_week", "horizon"] + chronos_cols].copy()
    chronos_pred_df = chronos_pred_df.merge(truth_df, on=["asin", "order_week", "horizon"], how="left")

    return predictor, chronos_pred_df


def load_exposure_data(data_raw, dph_cap_q=0.995):
    df = data_raw.copy()
    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    df = df.sort_values(["asin", "order_week"]).reset_index(drop=True)

    for c in ["fbi_demand", "total_dph", "buy_box_dph", "in_stock_dph"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    for c in ["total_dph", "buy_box_dph", "in_stock_dph"]:
        cap = df[c].quantile(dph_cap_q)
        df[c] = df[c].clip(upper=cap)

    if "our_price" in df.columns:
        df["our_price"] = pd.to_numeric(df["our_price"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["our_price"] = 0.0

    if "scot_oos" in df.columns:
        df["scot_oos"] = pd.to_numeric(df["scot_oos"], errors="coerce").fillna(0.0).clip(0, 1)
    else:
        df["scot_oos"] = 0.0

    df["order_month"] = df["order_week"].dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12.0)
    df["season_winter"] = df["order_month"].isin([12, 1, 2]).astype(float)
    df["season_spring"] = df["order_month"].isin([3, 4, 5]).astype(float)
    df["season_summer"] = df["order_month"].isin([6, 7, 8]).astype(float)
    df["season_fall"] = df["order_month"].isin([9, 10, 11]).astype(float)

    df, event_cols = add_explicit_event_features(df, week_col="order_week", event_window_weeks=4)
    df, static_cols = _encode_static_features(df)

    holiday_cols = [c for c in df.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in df.columns if c.startswith("distance_")]

    for c in holiday_cols + distance_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    context_cols = list(dict.fromkeys(
        ["our_price"]
        + holiday_cols
        + distance_cols
        + event_cols
        + [
            "order_month",
            "month_sin",
            "month_cos",
            "season_winter",
            "season_spring",
            "season_summer",
            "season_fall",
        ]
        + static_cols
        + [
            "hist_total_dph_last_log",
            "hist_total_dph_mean4_log",
            "hist_total_dph_mean13_log",
            "hist_buy_box_dph_last_log",
            "hist_buy_box_dph_mean4_log",
            "hist_buy_box_dph_mean13_log",
            "hist_instock_dph_last_log",
            "hist_instock_dph_mean4_log",
            "hist_instock_dph_mean13_log",
        ]
    ))

    for c in context_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    data = {}

    for asin, g in df.groupby("asin"):
        g = g.sort_values("order_week").reset_index(drop=True)

        demand = g["fbi_demand"].values.astype(np.float32)
        total = g["total_dph"].values.astype(np.float32)
        buy = g["buy_box_dph"].values.astype(np.float32)
        instock = g["in_stock_dph"].values.astype(np.float32)
        price = g["our_price"].values.astype(np.float32)
        oos = g["scot_oos"].values.astype(np.float32)

        week_idx = np.arange(len(g))

        features = np.stack(
            [
                np.log1p(demand),
                (demand > 0).astype(float),
                np.log1p(total),
                np.log1p(buy),
                np.log1p(instock),
                price,
                oos,
                np.sin(2 * np.pi * week_idx / 52.0),
                np.cos(2 * np.pi * week_idx / 52.0),
            ],
            axis=1,
        ).astype(np.float32)

        if np.std(features[:, 5]) > 1e-8:
            features[:, 5] = (features[:, 5] - features[:, 5].mean()) / (features[:, 5].std() + 1e-8)

        data[asin] = {
            "week": g["order_week"].values,
            "features": features,
            "demand": demand,
            "total_dph": total,
            "buy_box_dph": buy,
            "in_stock_dph": instock,
            "future_context": g[context_cols].values.astype(np.float32),
            "context_cols": context_cols,
        }

    print(f"TCN data ready | ASINs: {len(data)} | Context dim: {len(context_cols)}")
    return data, len(context_cols), context_cols


class ExposureDataset(Dataset):
    def __init__(self, data, history=52, horizon=20, mode="train", val_weeks=20, anchor_decay=0.08):
        self.samples = []
        self.data = data
        self.history = history
        self.horizon = horizon
        self.anchor_decay = anchor_decay

        for asin, d in data.items():
            T = len(d["features"])
            if mode == "train":
                starts = range(max(0, T - val_weeks - horizon - history + 1))
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []

            for start in starts:
                self.samples.append((asin, start))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _hist_mean(arr, end, window):
        x = arr[max(0, end - window):end]
        return float(np.mean(x)) if len(x) > 0 else 0.0

    def _make_future_context(self, d, start):
        h = self.history
        H = self.horizon
        fc = d["future_context"][start + h:start + h + H].copy()
        cols = d["context_cols"]
        idx = {c: i for i, c in enumerate(cols)}
        end = start + h

        total = d["total_dph"]
        buy = d["buy_box_dph"]
        instock = d["in_stock_dph"]

        for step_h in range(H):
            decay = np.exp(-self.anchor_decay * step_h)

            for prefix, arr in [("total", total), ("buy_box", buy), ("instock", instock)]:
                last_val = np.log1p(arr[end - 1]) if end > 0 else 0.0
                mean4_val = np.log1p(self._hist_mean(arr, end, 4))
                mean13_val = np.log1p(self._hist_mean(arr, end, 13))

                vals = {
                    f"hist_{prefix}_dph_last_log": decay * last_val + (1 - decay) * mean13_val,
                    f"hist_{prefix}_dph_mean4_log": decay * mean4_val + (1 - decay) * mean13_val,
                    f"hist_{prefix}_dph_mean13_log": mean13_val,
                }

                for col, val in vals.items():
                    if col in idx:
                        fc[step_h, idx[col]] = val

        return fc

    def __getitem__(self, i):
        asin, start = self.samples[i]
        d = self.data[asin]
        h = self.history
        H = self.horizon

        return {
            "asin": asin,
            "target_week": [str(w)[:10] for w in d["week"][start + h:start + h + H]],
            "x": torch.tensor(d["features"][start:start + h], dtype=torch.float32),
            "future_context": torch.tensor(self._make_future_context(d, start), dtype=torch.float32),
            "future_total_dph": torch.tensor(d["total_dph"][start + h:start + h + H], dtype=torch.float32),
            "future_buy_box_dph": torch.tensor(d["buy_box_dph"][start + h:start + h + H], dtype=torch.float32),
            "future_instock_dph": torch.tensor(d["in_stock_dph"][start + h:start + h + H], dtype=torch.float32),
            "future_demand": torch.tensor(d["demand"][start + h:start + h + H], dtype=torch.float32),
        }


def exposure_collate(batch):
    tensor_keys = [
        "x",
        "future_context",
        "future_total_dph",
        "future_buy_box_dph",
        "future_instock_dph",
        "future_demand",
    ]

    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in tensor_keys}
    out["asin"] = [b["asin"] for b in batch]
    out["target_week"] = [b["target_week"] for b in batch]
    return out


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class HistoryEncoderFull(nn.Module):
    def __init__(self, input_dim, d_model=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        dilations = [1, 2, 4, 8, 13, 26]
        self.convs = nn.ModuleList([CausalConv1d(d_model, d_model, kernel_size=2, dilation=d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.input_proj(x).transpose(1, 2)

        for conv, norm in zip(self.convs, self.norms):
            z = conv(h)
            h = h + z
            h = h.transpose(1, 2)
            h = norm(h)
            h = F.gelu(h)
            h = h.transpose(1, 2)

        return self.final_norm(h.transpose(1, 2))


class HorizonTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.10):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        z = x.transpose(1, 2)
        z = self.drop(F.relu(self.conv1(z)))
        z = self.drop(F.relu(self.conv2(z)))
        z = z.transpose(1, 2)
        m = min(z.shape[1], res.shape[1])
        return self.norm(res[:, :m, :] + z[:, :m, :])


class TCNDecoderWithCrossAttn(nn.Module):
    def __init__(self, d_model, context_dim, horizon=20, hidden=96, n_heads=4, dropout=0.10):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Sequential(nn.Linear(context_dim + 2, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.tcn = nn.ModuleList([
            HorizonTCNBlock(hidden, dilation=1, dropout=dropout),
            HorizonTCNBlock(hidden, dilation=2, dropout=dropout),
            HorizonTCNBlock(hidden, dilation=4, dropout=dropout),
        ])
        self.dec_proj = nn.Linear(hidden, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_norm = nn.LayerNorm(d_model)
        self.active_head = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.mag_head = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, enc_out, future_context, return_aux=False):
        B, H, _ = future_context.shape
        h_idx = torch.arange(H, device=future_context.device).float()
        h_norm = h_idx.view(1, H, 1).expand(B, H, 1) / max(H, 1)
        hsin = torch.sin(2 * torch.pi * h_norm)
        hcos = torch.cos(2 * torch.pi * h_norm)

        x = torch.cat([future_context, hsin, hcos], dim=-1)
        z = self.input_proj(x)

        for block in self.tcn:
            z = block(z)

        q = self.dec_proj(z)

        attn_out, attn_w = self.cross_attn(q, enc_out, enc_out, need_weights=return_aux)
        z_out = self.post_norm(q + attn_out)

        active_logit = self.active_head(z_out)
        p_active = torch.sigmoid(active_logit)
        log_mag = F.softplus(self.mag_head(z_out))
        log_hat = p_active * log_mag

        if return_aux:
            return {
                "log_hat": log_hat,
                "active_logit": active_logit,
                "p_active": p_active,
                "log_mag": log_mag,
                "attn_weights": attn_w,
            }

        return log_hat


class ExposureForecastModelV2(nn.Module):
    def __init__(self, input_dim, context_dim, d_model=64, horizon=20, n_heads=4, dropout=0.10):
        super().__init__()
        self.encoder = HistoryEncoderFull(input_dim=input_dim, d_model=d_model)
        self.decoder = TCNDecoderWithCrossAttn(
            d_model=d_model,
            context_dim=context_dim,
            horizon=horizon,
            hidden=max(96, d_model * 2),
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x, future_context, return_aux=False):
        enc_out = self.encoder(x)
        return self.decoder(enc_out, future_context, return_aux=return_aux)


def exposure_hurdle_loss(
    log_hat,
    true_total,
    true_buy,
    true_instock,
    active_logit,
    log_mag,
    w_total=0.30,
    w_buy=0.60,
    w_instock=1.00,
    bce_weight=1.00,
    mag_weight=1.00,
    mean_weight=0.20,
    horizon_weight_alpha=0.40,
    high_weight_alpha=1.00,
):
    true = torch.stack(
        [
            true_total.clamp(min=0.0),
            true_buy.clamp(min=0.0),
            true_instock.clamp(min=0.0),
        ],
        dim=-1,
    )

    target_log = torch.log1p(true)
    tw = torch.tensor([w_total, w_buy, w_instock], dtype=log_hat.dtype, device=log_hat.device).view(1, 1, 3)

    denom = target_log.detach().mean(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    high_w = 1.0 + high_weight_alpha * target_log.detach() / denom

    H = true.shape[1]
    h = torch.arange(1, H + 1, device=true.device, dtype=true.dtype).view(1, H, 1)
    horizon_w = 1.0 + horizon_weight_alpha * (h / max(float(H), 1.0))
    sample_w = high_w * horizon_w

    active_label = (true > 0).float()
    bce = F.binary_cross_entropy_with_logits(active_logit, active_label, reduction="none")
    bce_loss = (bce * sample_w * tw).mean()

    mag_err = F.huber_loss(log_mag, target_log, delta=1.0, reduction="none")
    mag_loss = (mag_err * active_label * sample_w * tw).sum() / (active_label * sample_w * tw).sum().clamp_min(1.0)

    pred_level = torch.expm1(log_hat).clamp(min=0.0)
    mean_pred = torch.log1p(pred_level.mean(dim=(0, 1)))
    mean_true = torch.log1p(true.mean(dim=(0, 1)).clamp_min(1e-6))
    mean_loss = (torch.abs(mean_pred - mean_true) * tw.view(3)).mean()

    return bce_weight * bce_loss + mag_weight * mag_loss + mean_weight * mean_loss


def train_exposure_model_v2(
    model,
    tr_ld,
    va_ld,
    epochs=60,
    lr=1e-3,
    patience=8,
    w_total=0.30,
    w_buy=0.60,
    w_instock=1.00,
    bce_weight=1.00,
    mag_weight=1.00,
    mean_weight=0.20,
    horizon_weight_alpha=0.40,
    high_weight_alpha=1.00,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_val = float("inf")
    best_sd = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        tr_sum = 0.0
        tr_n = 0

        for b in tr_ld:
            aux = model(b["x"], b["future_context"], return_aux=True)

            loss = exposure_hurdle_loss(
                log_hat=aux["log_hat"],
                true_total=b["future_total_dph"],
                true_buy=b["future_buy_box_dph"],
                true_instock=b["future_instock_dph"],
                active_logit=aux["active_logit"],
                log_mag=aux["log_mag"],
                w_total=w_total,
                w_buy=w_buy,
                w_instock=w_instock,
                bce_weight=bce_weight,
                mag_weight=mag_weight,
                mean_weight=mean_weight,
                horizon_weight_alpha=horizon_weight_alpha,
                high_weight_alpha=high_weight_alpha,
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += loss.item() * b["x"].shape[0]
            tr_n += b["x"].shape[0]

        sch.step()

        model.eval()
        va_sum = 0.0
        va_n = 0

        with torch.no_grad():
            for b in va_ld:
                aux = model(b["x"], b["future_context"], return_aux=True)

                loss = exposure_hurdle_loss(
                    log_hat=aux["log_hat"],
                    true_total=b["future_total_dph"],
                    true_buy=b["future_buy_box_dph"],
                    true_instock=b["future_instock_dph"],
                    active_logit=aux["active_logit"],
                    log_mag=aux["log_mag"],
                    w_total=w_total,
                    w_buy=w_buy,
                    w_instock=w_instock,
                    bce_weight=bce_weight,
                    mag_weight=mag_weight,
                    mean_weight=mean_weight,
                    horizon_weight_alpha=horizon_weight_alpha,
                    high_weight_alpha=high_weight_alpha,
                )

                va_sum += loss.item() * b["x"].shape[0]
                va_n += b["x"].shape[0]

        tr_loss = tr_sum / max(tr_n, 1)
        va_loss = va_sum / max(va_n, 1)

        print(f"Epoch {epoch + 1:03d} | train={tr_loss:.5f} | val={va_loss:.5f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch + 1}. Best val={best_val:.5f}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    return model


def predict_exposure_v2(model, va_ld, apply_funnel_constraint=True):
    rows = []
    model.eval()

    with torch.no_grad():
        for b in va_ld:
            aux = model(b["x"], b["future_context"], return_aux=True)
            pred = torch.expm1(aux["log_hat"]).clamp(min=0.0).cpu().numpy()
            pact = aux["p_active"].cpu().numpy()

            if apply_funnel_constraint:
                pred[:, :, 1] = np.minimum(pred[:, :, 1], pred[:, :, 0])
                pred[:, :, 2] = np.minimum(pred[:, :, 2], pred[:, :, 1])

            B, H = b["future_instock_dph"].shape

            for i in range(B):
                for h in range(H):
                    rows.append(
                        {
                            "asin": b["asin"][i],
                            "order_week": pd.to_datetime(b["target_week"][i][h]),
                            "horizon": h + 1,
                            "true_total_dph": b["future_total_dph"][i, h].item(),
                            "pred_total_dph": pred[i, h, 0],
                            "true_buy_box_dph": b["future_buy_box_dph"][i, h].item(),
                            "pred_buy_box_dph": pred[i, h, 1],
                            "true_instock_dph": b["future_instock_dph"][i, h].item(),
                            "pred_instock_dph": pred[i, h, 2],
                            "true_demand": b["future_demand"][i, h].item(),
                            "p_active_total": pact[i, h, 0],
                            "p_active_buy_box": pact[i, h, 1],
                            "p_active_instock": pact[i, h, 2],
                        }
                    )

    return pd.DataFrame(rows)


def exposure_metrics(pred_df, prefix="pred"):
    specs = [
        ("total_dph", "true_total_dph", f"{prefix}_total_dph"),
        ("buy_box_dph", "true_buy_box_dph", f"{prefix}_buy_box_dph"),
        ("in_stock_dph", "true_instock_dph", f"{prefix}_instock_dph"),
    ]

    rows = []

    for name, true_col, pred_col in specs:
        y = pred_df[true_col].values
        p = pred_df[pred_col].values

        rows.append(
            {
                "target": name,
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p),
                "pred_true_ratio": np.mean(p) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p),
                "corr": _corr(y, p),
                "active_AUC": _auc((y > 0).astype(int), p),
                "zero_rate_true": np.mean(y <= 0),
            }
        )

    return pd.DataFrame(rows)


def print_exposure_diagnostics(pred_df):
    print("=" * 100)
    print("TCN exposure metrics")
    print("=" * 100)

    model_tbl = exposure_metrics(pred_df, prefix="pred")
    print(model_tbl.round(5).to_string(index=False))

    rows = []

    for h, g in pred_df.groupby("horizon"):
        y = g["true_instock_dph"].values
        p = g["pred_instock_dph"].values

        rows.append(
            {
                "horizon": h,
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p),
                "ratio": np.mean(p) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p),
                "corr": _corr(y, p),
                "active_AUC": _auc((y > 0).astype(int), p),
            }
        )

    by_h = pd.DataFrame(rows)
    print("=" * 100)
    print("TCN by horizon: in_stock_dph")
    print("=" * 100)
    print(by_h.round(5).to_string(index=False))

    return {"model": model_tbl, "by_horizon": by_h}


def make_external_hat_df(pred_df):
    out = pred_df[["asin", "order_week", "pred_total_dph", "pred_buy_box_dph", "pred_instock_dph"]].copy()
    out["external_total_dph_hat_log"] = np.log1p(out["pred_total_dph"].clip(lower=0.0))
    out["external_buy_box_dph_hat_log"] = np.log1p(out["pred_buy_box_dph"].clip(lower=0.0))
    out["external_instock_dph_hat_log"] = np.log1p(out["pred_instock_dph"].clip(lower=0.0))
    return out


def run_exposure_v2(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
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
):
    print("=" * 100)
    print("Run TCN exposure model")
    print("=" * 100)

    df = prepare_data_from_sample_scot_intersection(data_raw1, scot_df, n_asins, seed)

    if remove_extreme:
        df = filter_extreme_asins(df, q=extreme_q)

    data, context_dim, context_cols = load_exposure_data(df, dph_cap_q=dph_cap_q)

    tr_ds = ExposureDataset(data, history=history, horizon=horizon, mode="train", val_weeks=horizon, anchor_decay=anchor_decay)
    va_ds = ExposureDataset(data, history=history, horizon=horizon, mode="val", val_weeks=horizon, anchor_decay=anchor_decay)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=exposure_collate)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=exposure_collate)

    print(f"Train samples: {len(tr_ds)} | Val samples: {len(va_ds)}")

    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise ValueError("TCN train/val dataset is empty")

    input_dim = next(iter(tr_ld))["x"].shape[-1]

    model = ExposureForecastModelV2(
        input_dim=input_dim,
        context_dim=context_dim,
        d_model=d_model,
        horizon=horizon,
        n_heads=n_heads,
        dropout=0.10,
    )

    print(f"Input dim: {input_dim} | Context dim: {context_dim}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model = train_exposure_model_v2(
        model=model,
        tr_ld=tr_ld,
        va_ld=va_ld,
        epochs=epochs,
        lr=lr,
        patience=patience,
        bce_weight=bce_weight,
        mag_weight=mag_weight,
        mean_weight=mean_weight,
        horizon_weight_alpha=horizon_weight_alpha,
        high_weight_alpha=high_weight_alpha,
    )

    pred_df = predict_exposure_v2(model, va_ld, apply_funnel_constraint=apply_funnel_constraint)
    diagnostics = print_exposure_diagnostics(pred_df)
    exposure_hat_for_demand = make_external_hat_df(pred_df)

    return {
        "model": model,
        "forecast_df": pred_df,
        "diagnostics": diagnostics,
        "exposure_hat_for_demand": exposure_hat_for_demand,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "context_cols": context_cols,
        "context_dim": context_dim,
        "data": data,
    }


def compare_chronos_vs_tcn(tcn_pred_df, chronos_pred_df, save_path=None):
    out = tcn_pred_df.copy()
    chr_df = chronos_pred_df.copy()

    out["asin"] = out["asin"].astype(str)
    chr_df["asin"] = chr_df["asin"].astype(str)

    out["order_week"] = pd.to_datetime(out["order_week"])
    chr_df["order_week"] = pd.to_datetime(chr_df["order_week"])

    compare_df = out.merge(chr_df, on=["asin", "order_week", "horizon"], how="inner")

    if len(compare_df) == 0:
        raise ValueError("compare_df is empty after merge")

    y = compare_df["true_instock_dph"].values.astype(float)
    p_tcn = compare_df["pred_instock_dph"].values.astype(float)
    p_chr = compare_df["chronos_pred_instock_dph"].values.astype(float)

    summary = pd.DataFrame(
        [
            {
                "model": "Your_TCN",
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p_tcn),
                "pred_true_ratio": np.mean(p_tcn) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p_tcn),
                "corr": _corr(y, p_tcn),
                "spearman": _safe_spearman(y, p_tcn),
                "active_AUC": _auc((y > 0).astype(int), p_tcn),
                "zero_rate_true": np.mean(y <= 0),
            },
            {
                "model": "Chronos2_FineTuned",
                "true_mean": np.mean(y),
                "pred_mean": np.mean(p_chr),
                "pred_true_ratio": np.mean(p_chr) / (np.mean(y) + 1e-8),
                "WAPE": _wape(y, p_chr),
                "corr": _corr(y, p_chr),
                "spearman": _safe_spearman(y, p_chr),
                "active_AUC": _auc((y > 0).astype(int), p_chr),
                "zero_rate_true": np.mean(y <= 0),
            },
        ]
    )

    rows = []

    for h, g in compare_df.groupby("horizon"):
        y_h = g["true_instock_dph"].values.astype(float)
        tcn_h = g["pred_instock_dph"].values.astype(float)
        chr_h = g["chronos_pred_instock_dph"].values.astype(float)

        rows.append(
            {
                "horizon": h,
                "true_mean": np.mean(y_h),
                "tcn_mean": np.mean(tcn_h),
                "tcn_ratio": np.mean(tcn_h) / (np.mean(y_h) + 1e-8),
                "tcn_WAPE": _wape(y_h, tcn_h),
                "tcn_corr": _corr(y_h, tcn_h),
                "chronos_mean": np.mean(chr_h),
                "chronos_ratio": np.mean(chr_h) / (np.mean(y_h) + 1e-8),
                "chronos_WAPE": _wape(y_h, chr_h),
                "chronos_corr": _corr(y_h, chr_h),
            }
        )

    by_horizon = pd.DataFrame(rows)

    compare_df["tcn_abs_err_instock"] = np.abs(compare_df["true_instock_dph"] - compare_df["pred_instock_dph"])
    compare_df["chronos_abs_err_instock"] = np.abs(compare_df["true_instock_dph"] - compare_df["chronos_pred_instock_dph"])
    compare_df["chronos_minus_tcn_abs_err"] = compare_df["chronos_abs_err_instock"] - compare_df["tcn_abs_err_instock"]
    compare_df["chronos_better"] = (compare_df["chronos_abs_err_instock"] < compare_df["tcn_abs_err_instock"]).astype(int)

    win_rate = compare_df["chronos_better"].mean()

    print("=" * 100)
    print("Final comparison")
    print(f"Merged rows: {len(compare_df)} | ASINs: {compare_df['asin'].nunique()} | Chronos better rate: {win_rate:.5f}")
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


def run_full_chronos_tcn_experiment(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
    history=52,
    horizon=20,
    chronos_main_quantile="0.5",
    use_glance_view_count=False,
    chronos_time_limit=14400,
    fine_tune_steps=2000,
    fine_tune_lr=1e-6,
    tcn_epochs=60,
    tcn_lr=1e-3,
    tcn_patience=8,
    ag_model_path="AutogluonModels/ag_InStockDPH_Chronos2_FineTuned",
    save_compare_path="chronos2_instock_compare_df.parquet",
):
    chronos_df, known_covariates = prepare_chronos_df(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
        dph_cap_q=0.995,
        remove_extreme=True,
        extreme_q=0.99,
        use_glance_view_count=use_glance_view_count,
    )

    chronos_predictor, chronos_pred_df = train_predict_chronos(
        df=chronos_df,
        known_covariates=known_covariates,
        history=history,
        horizon=horizon,
        main_quantile=chronos_main_quantile,
        ag_model_path=ag_model_path,
        time_limit=chronos_time_limit,
        fine_tune_steps=fine_tune_steps,
        fine_tune_lr=fine_tune_lr,
    )

    print("Chronos done:", chronos_pred_df.shape)

    result_tcn = run_exposure_v2(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=n_asins,
        seed=seed,
        history=history,
        horizon=horizon,
        d_model=64,
        n_heads=4,
        batch_size=64,
        epochs=tcn_epochs,
        lr=tcn_lr,
        patience=tcn_patience,
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

    comparison = compare_chronos_vs_tcn(
        tcn_pred_df=pred_df,
        chronos_pred_df=chronos_pred_df,
        save_path=save_compare_path,
    )

    result = {
        "chronos_predictor": chronos_predictor,
        "chronos_pred_df": chronos_pred_df,
        "known_covariates": known_covariates,
        "chronos_data": chronos_df,
        "result_tcn": result_tcn,
        "pred_df": pred_df,
        "comparison": comparison,
        "compare_df": comparison["compare_df"],
        "summary": comparison["summary"],
        "by_horizon": comparison["by_horizon"],
    }

    gc.collect()
    return result


N_ASINS = 5000
SEED = 42
HISTORY = 52
HORIZON = 20

CHRONOS_MAIN_QUANTILE = "0.5"
USE_GLANCE_VIEW_COUNT = False

CHRONOS_TIME_LIMIT = 14400
FINE_TUNE_STEPS = 2000
FINE_TUNE_LR = 1e-6

TCN_EPOCHS = 60
TCN_LR = 1e-3
TCN_PATIENCE = 8

result = run_full_chronos_tcn_experiment(
    data_raw1=data_raw1,
    scot_df=scot_df,
    n_asins=N_ASINS,
    seed=SEED,
    history=HISTORY,
    horizon=HORIZON,
    chronos_main_quantile=CHRONOS_MAIN_QUANTILE,
    use_glance_view_count=USE_GLANCE_VIEW_COUNT,
    chronos_time_limit=CHRONOS_TIME_LIMIT,
    fine_tune_steps=FINE_TUNE_STEPS,
    fine_tune_lr=FINE_TUNE_LR,
    tcn_epochs=TCN_EPOCHS,
    tcn_lr=TCN_LR,
    tcn_patience=TCN_PATIENCE,
)

chronos_pred_df = result["chronos_pred_df"]
pred_df = result["pred_df"]
compare_df = result["compare_df"]
summary = result["summary"]
by_horizon = result["by_horizon"]

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
