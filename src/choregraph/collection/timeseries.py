# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Time series transform functions — date extraction, rolling windows, lag/lead, datetime offset, and forecasting."""
import pandas as pd
import numpy as np
from typing import Optional, Union


def extract_date_part(df: pd.DataFrame, column: str, part: str,
                      output_column: Optional[str] = None) -> pd.DataFrame:
    """Extract a date/time component from a datetime column.

    Args:
        df: Input DataFrame.
        column: Column containing datetime values (auto-converted via
            ``pd.to_datetime``).
        part: Component to extract — ``"YEAR"``, ``"MONTH"``, ``"DAY"``,
            ``"DAY_OF_YEAR"``, or ``"WEEKDAY"``.
        output_column: Name of the result column (defaults to
            ``"{column}_{part_lower}"``).

    Returns:
        DataFrame with the extracted component column added.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()

    if column not in res_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    col_name = output_column if output_column else f"{column}_{part.lower()}"

    dt_col = pd.to_datetime(res_df[column])

    part_map = {
        "YEAR": dt_col.dt.year,
        "MONTH": dt_col.dt.month,
        "DAY": dt_col.dt.day,
        "DAY_OF_YEAR": dt_col.dt.day_of_year,
        "WEEKDAY": dt_col.dt.weekday,
    }

    part_upper = part.upper()
    if part_upper not in part_map:
        raise ValueError(f"Unsupported date part: {part}. "
                         f"Choose from {list(part_map.keys())}")

    res_df[col_name] = part_map[part_upper]
    return res_df


def rolling_statistics(df: pd.DataFrame, column: str,
                       window_size: Optional[int] = None,
                       function: str = "MEAN",
                       group_by: Optional[Union[str, list]] = None,
                       output_column: Optional[str] = None) -> pd.DataFrame:
    """Compute a rolling-window or cumulative statistic for a column.

    Replaces ``window_cumul`` — when *function* is ``"CUMSUM"`` the behaviour
    is identical (cumulative sum, *window_size* ignored).

    Rolling functions (MEAN, STD, MIN, MAX, SUM) automatically use a centered
    window so that smoothed values are temporally aligned with the source data.
    CUMSUM uses trailing accumulation by nature.

    Args:
        df: Input DataFrame.
        column: Column to compute the statistic on.
        window_size: Rolling window size (required for all functions except
            ``"CUMSUM"``).
        function: Aggregation function — ``"MEAN"``, ``"STD"``, ``"MIN"``,
            ``"MAX"``, ``"SUM"``, or ``"CUMSUM"``.
        group_by: Optional column(s) to group by before computing.
        output_column: Name of the result column (defaults to
            ``"{column}_{function_lower}"``).

    Returns:
        DataFrame with the computed column added.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()

    if column not in res_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    func_upper = function.upper()
    col_name = output_column if output_column else f"{column}_{func_upper.lower()}"

    gb_cols = None
    if group_by:
        gb_cols = [group_by] if isinstance(group_by, str) else group_by
        for c in gb_cols:
            if c not in res_df.columns:
                raise ValueError(f"Grouping column '{c}' not found in DataFrame")

    if func_upper == "CUMSUM":
        if gb_cols:
            res_df[col_name] = res_df.groupby(gb_cols)[column].cumsum()
        else:
            res_df[col_name] = res_df[column].cumsum()
    else:
        if window_size is None:
            raise ValueError(f"window_size is required for function '{function}'")
        window_size = int(window_size)

        rolling_funcs = {
            "MEAN": "mean",
            "STD": "std",
            "MIN": "min",
            "MAX": "max",
            "SUM": "sum",
        }

        if func_upper not in rolling_funcs:
            raise ValueError(f"Unsupported function: {function}. "
                             f"Choose from {list(rolling_funcs.keys()) + ['CUMSUM']}")

        pandas_method = rolling_funcs[func_upper]

        if gb_cols:
            res_df[col_name] = res_df.groupby(gb_cols)[column].transform(
                lambda x: getattr(x.rolling(window=window_size, center=True), pandas_method)()
            )
        else:
            res_df[col_name] = getattr(
                res_df[column].rolling(window=window_size, center=True), pandas_method
            )()

    return res_df


def lag_lead(df: pd.DataFrame, column: str, periods: int = 1,
             output_column: Optional[str] = None) -> pd.DataFrame:
    """Lag or lead a column by N periods.

    Args:
        df: Input DataFrame.
        column: Column to shift.
        periods: Number of periods to shift. Positive values lag (shift down),
            negative values lead (shift up).
        output_column: Name of the result column (defaults to
            ``"{column}_lag_{periods}"``).

    Returns:
        DataFrame with the shifted column added.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()

    if column not in res_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    periods = int(periods)
    col_name = output_column if output_column else f"{column}_lag_{periods}"

    res_df[col_name] = res_df[column].shift(periods)
    return res_df


def offset_datetime(df: pd.DataFrame, column: str, offset: int,
                    unit: str, output_column: Optional[str] = None) -> pd.DataFrame:
    """Add or subtract a time duration to every value in a datetime column.

    Args:
        df: Input DataFrame.
        column: Column containing datetime values (auto-converted via
            ``pd.to_datetime``).
        offset: Numeric value (positive = add, negative = subtract).
        unit: Time unit — ``"DAYS"``, ``"HOURS"``, ``"MINUTES"``,
            ``"SECONDS"``, ``"WEEKS"``, ``"MONTHS"``, or ``"YEARS"``.
        output_column: Name of the result column (defaults to
            ``"{column}_offset"``).

    Returns:
        DataFrame with the offset datetime column added.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    res_df = df.copy()

    if column not in res_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    offset = int(offset)
    col_name = output_column if output_column else f"{column}_offset"
    unit_upper = unit.upper()

    dt_col = pd.to_datetime(res_df[column])

    calendar_units = {
        "MONTHS": {"months": offset},
        "YEARS": {"years": offset},
    }
    timedelta_units = {
        "DAYS": "days",
        "HOURS": "hours",
        "MINUTES": "minutes",
        "SECONDS": "seconds",
        "WEEKS": "weeks",
    }

    if unit_upper in calendar_units:
        res_df[col_name] = dt_col + pd.DateOffset(**calendar_units[unit_upper])
    elif unit_upper in timedelta_units:
        res_df[col_name] = dt_col + pd.Timedelta(**{timedelta_units[unit_upper]: offset})
    else:
        raise ValueError(
            f"Unsupported unit: {unit}. "
            f"Choose from {list(timedelta_units.keys()) + list(calendar_units.keys())}"
        )

    return res_df


def forecast_time_series(df: pd.DataFrame, date_col: str, value_col: str,
                         horizon: int = 30,
                         perform_split_test: bool = False) -> pd.DataFrame:
    """Forecast a time series using RandomForest on pct-change features with
    Monte-Carlo recursive prediction and confidence intervals.

    Args:
        df: Input DataFrame with at least a date column and a numeric value column.
        date_col: Name of the column containing dates.
        value_col: Name of the column containing the numeric signal to forecast.
        horizon: Number of future steps to forecast (default 30).
        perform_split_test: If True, run an 80/20 backtest instead of future
            forecasting (dev/test only — not exposed in XSD).

    Returns:
        DataFrame with forecast results. In normal mode: ``Date``,
        ``{value_col}``, ``Forecast``, ``Forecast_Lower``, ``Forecast_Upper``.
        In split-test mode: ``Date``, ``Actual``, ``Predicted``.
    """
    # ------------------------------------------------------------------
    # Step 0 — Validation
    # ------------------------------------------------------------------
    if df is None or df.empty:
        return pd.DataFrame()

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in DataFrame")

    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # ------------------------------------------------------------------
    # Step 1 — Smart Frequency Inference & Sanitization
    # ------------------------------------------------------------------
    res = df[[date_col, value_col]].copy()
    res[date_col] = pd.to_datetime(res[date_col], dayfirst=True)
    res[value_col] = pd.to_numeric(res[value_col], errors='coerce')
    res = res.dropna(subset=[date_col, value_col])
    res = res.sort_values(date_col).reset_index(drop=True)

    # Infer the natural frequency from the modal gap
    gaps = res[date_col].diff().dropna()
    if gaps.empty:
        raise ValueError("Not enough data points to infer frequency")

    modal_gap = gaps.mode().iloc[0]

    # Decide resampling strategy
    res = res.set_index(date_col)

    if modal_gap >= pd.Timedelta(days=1):
        # Daily-scale data — check if missing days are mostly weekends
        all_gaps_days = gaps.dt.days
        if modal_gap <= pd.Timedelta(days=2):
            # Check weekend pattern: generate full calendar range and see
            # which missing dates fall on Sat/Sun
            full_range = pd.date_range(res.index.min(), res.index.max(), freq='D')
            missing_dates = full_range.difference(res.index)
            if len(missing_dates) > 0:
                weekend_missing = sum(d.weekday() >= 5 for d in missing_dates)
                weekend_ratio = weekend_missing / len(missing_dates)
            else:
                weekend_ratio = 0.0

            if weekend_ratio > 0.6:
                res = res.resample('B').ffill()
            else:
                res = res.resample('D').interpolate()
        else:
            # Larger gaps (weekly, monthly, etc.) — resample to modal gap
            res = res.resample(modal_gap).interpolate()
    else:
        # Sub-daily data — resample to the modal gap
        res = res.resample(modal_gap).interpolate()

    # Drop any remaining NaN from resampling edges
    res = res.dropna()

    # Data guard: cap at 5000 rows (keep most recent)
    if len(res) > 5000:
        res = res.iloc[-5000:]

    if len(res) < 25:
        raise ValueError(
            f"Insufficient data: need at least 25 rows after sanitization, got {len(res)}"
        )

    # ------------------------------------------------------------------
    # Step 2 — Signal extraction (pct_change for stationarity)
    # ------------------------------------------------------------------
    values = res[value_col].values.astype(float).copy()
    pct = res[value_col].pct_change().values.copy()
    pct[0] = 0.0
    pct = np.where(np.isinf(pct), 0.0, pct)
    pct = np.nan_to_num(pct, nan=0.0)

    # ------------------------------------------------------------------
    # Step 3 — Feature engineering
    # ------------------------------------------------------------------
    lag_steps = [1, 5, 10, 20]
    window_sizes = [5, 20]

    def _build_features(pct_series: np.ndarray) -> pd.DataFrame:
        s = pd.Series(pct_series)
        feat = pd.DataFrame()
        for lag in lag_steps:
            feat[f'lag_{lag}'] = s.shift(lag)
        for w in window_sizes:
            feat[f'rolling_mean_{w}'] = s.rolling(w).mean()
            feat[f'rolling_std_{w}'] = s.rolling(w).std()
        return feat

    features = _build_features(pct)
    features['target'] = pct

    # Drop rows where lags/rolling produce NaN (first ~20 rows)
    features = features.dropna(subset=[c for c in features.columns if c != 'target'
                                        and 'std' not in c])
    features = features.fillna(0.0)
    valid_start = features.index[0]

    X = features.drop(columns=['target']).values
    y = features['target'].values

    # Align values array to the same valid range
    values_aligned = values[valid_start:]

    # ------------------------------------------------------------------
    # Step 4 — Model training
    # ------------------------------------------------------------------
    from choregraph._extras import optional_dep
    with optional_dep():
        from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42,
        n_jobs=-1, oob_score=True
    )
    model.fit(X, y)

    # OOB residual sigma for uncertainty estimation
    model_sigma = np.std(y - model.oob_prediction_)

    # ------------------------------------------------------------------
    # Step 5/6 — Forecasting or Split-test
    # ------------------------------------------------------------------
    dates = res.index[valid_start:]

    if perform_split_test:
        # ---- Split-test mode: 80/20 backtest ----
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model_st = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42,
            n_jobs=-1, oob_score=True
        )
        model_st.fit(X_train, y_train)
        y_pred = model_st.predict(X_test)

        # Reconstruct absolute values from predicted pct via cumulative product
        base_val = values_aligned[split_idx - 1]
        predicted_vals = base_val * np.cumprod(1 + y_pred)

        # Return full history so the split point is visible on the graph
        all_dates = dates
        all_actual = values_aligned
        all_predicted = np.full(len(values_aligned), np.nan)
        all_predicted[split_idx:] = predicted_vals

        return pd.DataFrame({
            'Date': all_dates,
            'Actual': all_actual,
            'Predicted': all_predicted,
        }).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 5 — Vectorized path-dependent Monte-Carlo forecasting
    # ------------------------------------------------------------------
    N_SIMS = 100
    LOOKBACK = max(max(lag_steps), max(window_sizes))  # 20
    n_features = len(lag_steps) + 2 * len(window_sizes)

    last_val = values[-1]

    # Seed each simulation with the same trailing pct history
    seed_pct = pct[-LOOKBACK:]
    current_paths = np.tile(seed_pct, (N_SIMS, 1))  # (N_SIMS, LOOKBACK)

    sim_last_vals = np.full(N_SIMS, last_val)
    sim_values = np.full((horizon, N_SIMS), np.nan)

    for t in range(horizon):
        # Build features for all simulations at once
        features_batch = np.zeros((N_SIMS, n_features))
        col = 0
        for lag in lag_steps:
            features_batch[:, col] = current_paths[:, -lag]
            col += 1
        for w in window_sizes:
            window_slice = current_paths[:, -w:]       # (N_SIMS, w)
            features_batch[:, col] = window_slice.mean(axis=1)
            col += 1
            if w > 1:
                features_batch[:, col] = window_slice.std(axis=1, ddof=1)
            col += 1

        features_batch = np.nan_to_num(features_batch, nan=0.0)

        # Batch prediction — each sim gets a different prediction
        preds = model.predict(features_batch)          # (N_SIMS,)

        # Add stochastic noise
        noise = np.random.normal(0, model_sigma, N_SIMS)
        noisy_pcts = preds + noise

        # Update absolute value trajectories
        sim_last_vals = sim_last_vals * (1 + noisy_pcts)
        sim_values[t] = sim_last_vals

        # Append new pct and trim to LOOKBACK to bound memory
        current_paths = np.hstack([current_paths, noisy_pcts.reshape(-1, 1)])
        if current_paths.shape[1] > LOOKBACK:
            current_paths = current_paths[:, -LOOKBACK:]

    # Aggregate across simulations
    forecast_median = np.median(sim_values, axis=1)
    forecast_lower = np.percentile(sim_values, 5, axis=1)
    forecast_upper = np.percentile(sim_values, 95, axis=1)

    # Build future dates
    last_date = res.index[-1]
    freq = pd.infer_freq(res.index[-30:]) if len(res) >= 30 else pd.infer_freq(res.index)
    if freq is None:
        freq = modal_gap
    future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                                 periods=horizon, freq=freq)

    # Build output: unified value column, Is_Forecast distinguishes real vs predicted
    hist_df = pd.DataFrame({
        'Date': dates,
        value_col: values_aligned,
        'Forecast_Lower': np.nan,
        'Forecast_Upper': np.nan,
        'Is_Forecast': False,
    })

    fc_df = pd.DataFrame({
        'Date': future_dates,
        value_col: forecast_median,
        'Forecast_Lower': forecast_lower,
        'Forecast_Upper': forecast_upper,
        'Is_Forecast': True,
    })

    result = pd.concat([hist_df, fc_df], ignore_index=True)
    return result
