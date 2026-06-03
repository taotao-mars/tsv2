for HISTORY in [13, 26, 52]:
    result_roll = run_exposure_v2_rolling(
        data_raw1=data_raw1,
        scot_df=scot_df,
        n_asins=1000,
        seed=42,
        history=HISTORY,
        horizon=20,
        rolling_offsets=(60, 40, 20, 0),
        epochs=20,
        patience=5,
        batch_size=128,
        use_scot_intersection=True,
        use_encoder_self_attn=True,
    )
    display(result_roll["rolling_summary"].assign(history=HISTORY).round(4))
