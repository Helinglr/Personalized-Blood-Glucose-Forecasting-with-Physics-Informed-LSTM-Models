import pandas as pd
import numpy as np

TIME_COL = "timestamp"
GLUCOSE_COL = "glucose"
BOLUS_COL = "bolus"
CARBS_COL = "carbs"

# -----------------------------
# Helpers
# -----------------------------
def get_slot(ts):
    h = ts.hour
    if 6 <= h < 11:
        return "Morning"
    elif 11 <= h < 16:
        return "Afternoon"
    elif 16 <= h < 23:
        return "Evening"
    else:
        return "Night"

def normalize_columns(df):
    rename_map = {
        "cbg": "glucose",
        "CGM": "glucose",
        "carbInput": "carbs",
        "carb": "carbs",
        "insulin": "bolus"
    }
    return df.rename(columns=rename_map)

def ensure_timestamp(df):
    if TIME_COL not in df.columns:
        print("⚠️ 'TIMESTAMP' column missing, generating synthetic timestamps.")
        df[TIME_COL] = pd.date_range(
            start="2022-01-01",
            periods=len(df),
            freq="5min"
        )
    else:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df

# -----------------------------
# Pure Insulin Events (ISF)
# -----------------------------
def extract_pure_insulin_events(df, min_bolus=0.5, window=120):
    df = df.sort_values(TIME_COL).copy()

    events = []

    for idx, row in df.iterrows():
        if row[BOLUS_COL] < min_bolus:
            continue

        t0 = row[TIME_COL]

        # +/- No carbohydrates should be present within the window.
        mask_carb = (
            (df[TIME_COL] >= t0 - pd.Timedelta(minutes=window)) &
            (df[TIME_COL] <= t0 + pd.Timedelta(minutes=window))
        )
        if df.loc[mask_carb, CARBS_COL].sum() > 0:
            continue

        # future glucose
        mask_future = (
            (df[TIME_COL] >= t0 + pd.Timedelta(minutes=window))
        )
        if mask_future.sum() == 0:
            continue

        g1 = df.loc[mask_future].iloc[0][GLUCOSE_COL]
        g0 = row[GLUCOSE_COL]
        delta_g = g1 - g0

        if delta_g >= 0:
            continue

        isf_emp = abs(delta_g) / row[BOLUS_COL]

        events.append({
            "timestamp": t0,
            "slot": get_slot(t0),
            "bolus": row[BOLUS_COL],
            "g0": g0,
            "g1": g1,
            "delta_g": delta_g,
            "ISF_empirical": isf_emp
        })

    return pd.DataFrame(events)

# -----------------------------
# Pure Carb Events (CS)
# -----------------------------
def extract_pure_carb_events(df, min_carbs=15, window=120):
    df = df.sort_values(TIME_COL).copy()
    events = []

    for idx, row in df.iterrows():
        carbs = row[CARBS_COL]
        if carbs < min_carbs:
            continue

        t0 = row[TIME_COL]

        # +/- No insulin should be present within the window.
        mask_bolus = (
            (df[TIME_COL] >= t0 - pd.Timedelta(minutes=window)) &
            (df[TIME_COL] <= t0 + pd.Timedelta(minutes=window))
        )
        if df.loc[mask_bolus, BOLUS_COL].sum() > 0:
            continue

        mask_future = (
            (df[TIME_COL] >= t0 + pd.Timedelta(minutes=window))
        )
        if mask_future.sum() == 0:
            continue

        g0 = row[GLUCOSE_COL]
        g1 = df.loc[mask_future].iloc[0][GLUCOSE_COL]
        delta_g = g1 - g0

        cs_emp = delta_g / carbs

        events.append({
            "timestamp": t0,
            "slot": get_slot(t0),
            "carbs": carbs,
            "g0": g0,
            "g1": g1,
            "delta_g": delta_g,
            "CS_empirical": cs_emp
        })

    return pd.DataFrame(events)


def summarize_with_iqr(df, value_col):
    """Clinically cleans noisy data and returns the median."""
    if df.empty: return "No Data"
    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    iqr = q3 - q1
    filtered = df[(df[value_col] >= q1 - 1.5*iqr) & (df[value_col] <= q3 + 1.5*iqr)]
    return filtered[value_col].median()

# -----------------------------
# Slot summary
# -----------------------------
def summarize_by_slot(df, value_col):
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby("slot")[value_col]
          .agg(["count", "median", "mean", "std"])
          .reset_index()
          .sort_values("slot")
    )

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("📂 Loading CSV...")
    df = pd.read_csv("data/540_ws_training_processed.csv")

    df = normalize_columns(df)
    df = ensure_timestamp(df)

    print("🧾 Columns:", df.columns.tolist())

    # Safety: missing columns
    for col in [GLUCOSE_COL, BOLUS_COL, CARBS_COL]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_values(TIME_COL)

    print("🔬 Extracting ISF events...")
    # validator.py içindeki çağrı:
    isf_events = extract_pure_insulin_events_optimized(
    df, 
    min_bolus=0.3,       # 0.5 -> 0.3 (Capture smaller corrections)
    window=180,          # 120 -> 180 (Capture insulin effect tail)
    slope_threshold=0.2  # Initial glucose should not be too volatile
)

    print("🍞 Extracting CS events...")
    cs_events = extract_pure_carb_events(df)

    print("\n=== PURE INSULIN (ISF) SUMMARY ===")
    print(summarize_by_slot(isf_events, "ISF_empirical"))

    print("\n=== PURE CARB (CS) SUMMARY ===")
    print(summarize_by_slot(cs_events, "CS_empirical"))

    isf_events.to_csv("patient_540_pure_isf_events.csv", index=False)
    cs_events.to_csv("patient_540_pure_cs_events.csv", index=False)

    print("\n✅ Clinical validation files generated:")
    print("   - patient_540_pure_isf_events.csv")
    print("   - patient_540_pure_cs_events.csv")
