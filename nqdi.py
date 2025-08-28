# app.py
"""
nQDI™ Live — HR-dominant + Normalized-Power-style + Best/Worst nQDI windows
Run:
    pip install streamlit plotly pandas numpy fitparse
    streamlit run app.py
"""
import io, math, random
import numpy as np
import pandas as pd
import streamlit as st
from fitparse import FitFile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------
# Helpers
# ---------------------------

st.set_page_config(page_title="Normalized Quad Deficit Index™ (nQDI™) Live Calculator", layout="wide")

def parse_fit_bytes(b: bytes, speed_thresh=0.5):
    """Parse FIT bytes to dataframe with time,power,hr,speed. Resample to 1s. Remove stops if speed present."""
    fit = FitFile(io.BytesIO(b))
    rows = []
    for msg in fit.get_messages("record"):
        rec = {"time": None, "power": None, "hr": None, "speed": None}
        for d in msg:
            if d.name == "timestamp":
                rec["time"] = pd.to_datetime(d.value)
            elif d.name == "power":
                rec["power"] = d.value
            elif d.name == "heart_rate":
                rec["hr"] = d.value
            elif d.name == "speed":
                rec["speed"] = d.value
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["time","power","hr","speed"])
    # If timestamps exist, use them and resample to 1s
    if df["time"].notna().any():
        df = df.dropna(subset=["power"]).copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        df = df.resample("1S").mean()
        df["hr"] = df["hr"].ffill()
        df["speed"] = df["speed"].ffill().fillna(0)
        df = df.reset_index()
    else:
        df = df.dropna(subset=["power"]).reset_index(drop=True)
        df["time"] = pd.to_timedelta(df.index, unit="s")
        df["hr"] = df["hr"].ffill().fillna(0)
        df["speed"] = df.get("speed", pd.Series(0,index=df.index)).fillna(0)
    # numeric
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    # Remove stops if speed data meaningful
    if df["speed"].max() > 0:
        filtered = df[df["speed"] > speed_thresh].reset_index(drop=True)
        if len(filtered) == 0:
            # if everything removed, keep original but notify later
            return df
        else:
            return filtered
    return df

def rolling_np_ts(power_arr, window=30):
    """Return per-second NP-style series: rolling window of (mean(x**4))**0.25."""
    s = pd.Series(power_arr.astype(float))
    def window_np(x):
        if x.size == 0:
            return 0.0
        return float(np.mean(np.power(x, 4)) ** 0.25)
    roll = s.rolling(window=window, min_periods=1).apply(lambda x: window_np(x), raw=True)
    return roll.to_numpy()

def scalar_np(power_arr, window=30):
    """Scalar NP over entire ride."""
    s = pd.Series(power_arr.astype(float))
    if len(s) < window:
        return float(s.mean()) if len(s) else 0.0
    roll_mean = s.rolling(window=window, min_periods=window).mean().dropna()
    if roll_mean.empty:
        return float(s.mean())
    return float(np.mean(np.power(roll_mean, 4)) ** 0.25)

def estimate_hr_from_power(power_arr, lthr, window=30, eps=1e-6, low_factor=0.6, high_factor=1.4):
    """Estimate per-second HR from power so identical files produce identical fallback estimates."""
    if lthr <= 0:
        return np.ones_like(power_arr) * lthr
    np_ts = rolling_np_ts(power_arr, window=window)
    # ratio: np_ts / power -> if power > np_ts -> ratio < 1 -> hr_est < lthr (dominance)
    ratio = np_ts / (power_arr + eps)
    hr_est = lthr * ratio
    hr_est = np.clip(hr_est, low_factor * lthr, high_factor * lthr)
    return hr_est

def effective_watts_hr_dominant(power_arr, hr_arr, lthr, alpha=1.5, clip_low=0.6, clip_high=1.5):
    """Lower HR -> dominance (>1) -> effective watts increase."""
    power = np.array(power_arr, dtype=float)
    hr = np.array(hr_arr, dtype=float)
    # if hr entries are zero or nan, hr_safe will be lthr (we'll usually estimate before calling if we need)
    hr_safe = np.where((hr <= 0) | np.isnan(hr), lthr if lthr and lthr>0 else 1.0, hr)
    dominance = np.divide(lthr, hr_safe, out=np.ones_like(hr_safe, dtype=float), where=(hr_safe!=0))
    dominance = np.clip(dominance, clip_low, clip_high)
    return power * (dominance ** alpha)

def compute_best_worst_windows(nqdi_series, secs, durations_secs):
    """For each duration (sec) compute min (best for you) and max (worst for you) rolling mean window.
       Return list of dicts with duration, best_val, best_idx, worst_val, worst_idx."""
    out = []
    arr = np.array(nqdi_series, dtype=float)
    N = len(arr)
    for d in durations_secs:
        if d <= 0 or d > N:
            out.append({"duration": d, "best": None, "best_idx": None, "worst": None, "worst_idx": None})
            continue
        # rolling mean of window d
        mat = pd.Series(arr).rolling(window=d, min_periods=d).mean().dropna().to_numpy()
        if mat.size == 0:
            out.append({"duration": d, "best": None, "best_idx": None, "worst": None, "worst_idx": None})
            continue
        best_val = float(np.min(mat))
        worst_val = float(np.max(mat))
        # indices in terms of end index of window
        best_end = int(np.argmin(mat) + (d-1))
        worst_end = int(np.argmax(mat) + (d-1))
        # map to secs array if available
        best_idx = int(best_end) if len(secs)>best_end else None
        worst_idx = int(worst_end) if len(secs)>worst_end else None
        out.append({"duration": d, "best": best_val, "best_idx": best_idx, "worst": worst_val, "worst_idx": worst_idx})
    return out

# ---------------------------
# UI
# ---------------------------

st.markdown("# 🚴‍♂️ Normalized Quad Deficit Index™ (nQDI™) Live Calculator")
st.caption("Powered by power meters, Strava grief, KOM delusions, and the Quad Deficit Index™ (QDI™).")

colA, colB = st.columns(2)
with colA:
    you_file = st.file_uploader("YOUR .fit file", type=["fit"], key="you_file")
    you_weight_str = st.text_input("Your weight (number only)", value="", key="you_weight")
    you_lthr_str = st.text_input("Your LTHR (bpm) — optional (leave blank if none)", value="", key="you_lthr")
with colB:
    friend_file = st.file_uploader("FRIEND .fit file", type=["fit"], key="friend_file")
    friend_weight_str = st.text_input("Friend weight (number only)", value="", key="friend_weight")
    friend_lthr_str = st.text_input("Friend LTHR (bpm) — optional (leave blank if none)", value="", key="friend_lthr")

unit = st.selectbox("Weight unit (applies to both)", ["kg","lbs"], index=0, key="weight_unit")
alpha = st.slider("HR exponent α (higher = HR matters more)", 0.5, 2.5, 1.5, 0.1, key="alpha")
smooth = st.slider("EMA smoothing of effective watts (seconds)", 1, 60, 20, 1, key="smooth")
np_window = st.slider("Rolling NP window (seconds) — surge sensitivity", 10, 60, 30, 1, key="np_window")
dur_options = st.multiselect("Best/Worst windows to compute (seconds) — choose some common durations:",
                             options=[15,30,60,120,300,600,1200,1800,3600],
                             default=[15,30,60,300,1200], key="dur_select")
compute = st.button("⚡ Compute nQDI™ Live (ruin a friendship)")

# ---------------------------
# Compute
# ---------------------------
if compute:
    # validate numeric inputs
    try:
        you_weight = float(you_weight_str)
        friend_weight = float(friend_weight_str)
    except Exception:
        st.error("Enter numeric weights for both riders (no units).")
        st.stop()
    # convert if needed
    if unit == "lbs":
        you_weight *= 0.45359237
        friend_weight *= 0.45359237
    # parse LTHR if provided
    try:
        you_lthr = float(you_lthr_str) if you_lthr_str.strip() != "" else None
    except:
        st.error("Invalid Your LTHR value.")
        st.stop()
    try:
        friend_lthr = float(friend_lthr_str) if friend_lthr_str.strip() != "" else None
    except:
        st.error("Invalid Friend LTHR value.")
        st.stop()
    # check files
    if not you_file or not friend_file:
        st.error("Upload both .fit files (you + friend).")
        st.stop()

    # read files
    try:
        you_df = parse_fit_bytes(you_file.read())
        friend_df = parse_fit_bytes(friend_file.read())
    except Exception as e:
        st.error(f"Failed to parse FIT files: {e}")
        st.stop()

    if you_df.empty or friend_df.empty:
        st.error("One ride had no power data after parsing.")
        st.stop()

    # detect HR presence (any HR > 0)
    hr_present_you = (you_df["hr"] > 0).any()
    hr_present_friend = (friend_df["hr"] > 0).any()

    # fallback HR estimation only if necessary
    fallback_msgs = []
    if not hr_present_you:
        if you_lthr is None:
            st.warning("YOU: No HR and no LTHR provided — using raw power NP (no HR adjustment).")
            # we'll handle by setting hr_est to zeros and later effective calculation will use hr_safe=lthr => might use lthr None fallback
        else:
            est = estimate_hr_from_power(you_df["power"].to_numpy(), you_lthr, window=np_window)
            you_df["hr"] = est
            fallback_msgs.append("YOU: HR missing — estimated HR from power.")
    if not hr_present_friend:
        if friend_lthr is None:
            st.warning("FRIEND: No HR and no LTHR provided — using raw power NP (no HR adjustment).")
        else:
            est = estimate_hr_from_power(friend_df["power"].to_numpy(), friend_lthr, window=np_window)
            friend_df["hr"] = est
            fallback_msgs.append("FRIEND: HR missing — estimated HR from power.")
    if fallback_msgs:
        st.info("  \n".join(fallback_msgs))

    # align by integer second from each start
    def add_sec_col(df):
        if np.issubdtype(df["time"].dtype, np.datetime64):
            return (df["time"] - df["time"].iloc[0]).dt.total_seconds().round().astype(int)
        else:
            return df["time"].dt.total_seconds().round().astype(int)
    you_df = you_df.reset_index(drop=True).copy()
    friend_df = friend_df.reset_index(drop=True).copy()
    you_df["sec"] = add_sec_col(you_df)
    friend_df["sec"] = add_sec_col(friend_df)
    merged = pd.merge(you_df, friend_df, on="sec", suffixes=("_you","_friend"))
    if merged.empty:
        st.error("No overlapping seconds after alignment. Try rides of similar duration or ensure speed/time present.")
        st.stop()

    # effective watts (HR-dominant inverted)
    y_eff = effective_watts_hr_dominant(merged["power_you"].to_numpy(),
                                       merged["hr_you"].to_numpy(),
                                       you_lthr if you_lthr else merged["hr_you"].mean() if hr_present_you else 1.0,
                                       alpha=alpha)
    f_eff = effective_watts_hr_dominant(merged["power_friend"].to_numpy(),
                                       merged["hr_friend"].to_numpy(),
                                       friend_lthr if friend_lthr else merged["hr_friend"].mean() if hr_present_friend else 1.0,
                                       alpha=alpha)

    # EMA smooth effective watts
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()
    y_eff_sm = ema(y_eff, span=smooth)
    f_eff_sm = ema(f_eff, span=smooth)

    # NP-style series on effective watts
    y_np_ts = rolling_np_ts(y_eff_sm, window=np_window)
    f_np_ts = rolling_np_ts(f_eff_sm, window=np_window)

    eps = 1e-6
    nqdi_sec = np.divide(f_np_ts, np.maximum(y_np_ts, eps))
    nqdi_sec = np.clip(nqdi_sec, 0.0, 10.0)

    # W/kg variant (on NP-style effective watts)
    y_np_wkg = y_np_ts / max(you_weight, 0.1)
    f_np_wkg = f_np_ts / max(friend_weight, 0.1)
    nqdi_sec_kg = np.divide(f_np_wkg, np.maximum(y_np_wkg, eps))
    nqdi_sec_kg = np.clip(nqdi_sec_kg, 0.0, 10.0)

    # summaries
    mean_nqdi = float(np.mean(nqdi_sec))
    mean_nqdi_kg = float(np.mean(nqdi_sec_kg))
    p95 = float(np.percentile(nqdi_sec, 95))

    scalar_y_np = scalar_np(y_eff_sm, window=np_window)
    scalar_f_np = scalar_np(f_eff_sm, window=np_window)

    # compute best/worst windows
    durations = sorted(list(set(dur_options if isinstance(dur_options, list) and dur_options else dur_options)))
    bw = compute_best_worst_windows(nqdi_sec, merged["sec"].to_numpy(), durations)

    # roast selection sets (tiered)
    LIGHT = ["Cute. Basically FTP twins.", "Close race — smiles all round. Post immediately."]
    MED = ["You got nudged. Admit nothing.", "Their KOM shelf just grew a tiny plaque with your name on it."]
    DARK = ["You were turned into a domestic domestique. Cry softly into your bidon.", "This hurts in a way that can't be fixed with tubeless sealant."]
    NUKE = ["Absolute annihilation. Sell your bike, take up pottery.", "Your QDI is a natural disaster. Evacuate your local group chat."]

    def pick_roast(q):
        if q is None or math.isnan(q):
            return "nQDI undefined — something's broken (or you were standing still and staring at the scenery)."
        if q < 0.95:
            return random.choice(LIGHT)
        elif q < 1.1:
            return random.choice(MED)
        elif q < 1.4:
            return random.choice(DARK)
        else:
            return random.choice(NUKE)

    roast_msg = pick_roast(mean_nqdi_kg)

    # ---------------------------
    # Output metrics & chart
    # ---------------------------
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Raw mean nQDI", f"{mean_nqdi:.3f}")
    c2.metric("Mean nQDI (W/kg)", f"{mean_nqdi_kg:.3f}")
    c3.metric("95th pct nQDI", f"{p95:.3f}")
    c4.metric("Your NP (eff)", f"{scalar_y_np:.0f} W")

    st.markdown(f"**You** — NP-style eff: **{scalar_y_np:.0f} W**, avg HR: **{merged['hr_you'].mean():.0f} bpm**, weight: **{you_weight:.1f} {unit}**  ")
    st.markdown(f"**Friend** — NP-style eff: **{scalar_f_np:.0f} W**, avg HR: **{merged['hr_friend'].mean():.0f} bpm**, weight: **{friend_weight:.1f} {unit}**  ")

    # plotly
    secs = merged["sec"].to_numpy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{}]],
                        subplot_titles=("Power & HR", "NP-style effective watts", "nQDI per second (raw & W/kg)"))

    fig.add_trace(go.Scatter(x=secs, y=merged["power_you"], name="You Power (W)", line=dict(color="blue")), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=secs, y=merged["power_friend"], name="Friend Power (W)", line=dict(color="red")), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=secs, y=merged["hr_you"], name="You HR (bpm)", line=dict(color="blue", dash="dot")), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=secs, y=merged["hr_friend"], name="Friend HR (bpm)", line=dict(color="red", dash="dot")), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=secs, y=y_np_ts, name="You NP-style eff W", line=dict(color="blue", dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=secs, y=f_np_ts, name="Friend NP-style eff W", line=dict(color="red", dash="dash")), row=2, col=1)

    fig.add_trace(go.Scatter(x=secs, y=nqdi_sec, name="nQDI raw", line=dict(color="black")), row=3, col=1)
    fig.add_trace(go.Scatter(x=secs, y=nqdi_sec_kg, name="nQDI W/kg", line=dict(color="orange")), row=3, col=1)

    # annotate best/worst windows
    for item in bw:
        if item["best"] is not None and item["best_idx"] is not None:
            idx = item["best_idx"]
            x = secs[idx]
            fig.add_vline(x=x, line=dict(color="green", width=1), row=3, col=1)
            fig.add_annotation(x=x, y=item["best"], text=f"BEST {item['duration']}s {item['best']:.2f}", showarrow=True, yshift=10, arrowcolor="green", row=3, col=1)
        if item["worst"] is not None and item["worst_idx"] is not None:
            idx = item["worst_idx"]
            x = secs[idx]
            fig.add_vline(x=x, line=dict(color="red", width=1), row=3, col=1)
            fig.add_annotation(x=x, y=item["worst"], text=f"WORST {item['duration']}s {item['worst']:.2f}", showarrow=True, yshift=-10, arrowcolor="red", row=3, col=1)

    fig.update_layout(height=900, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      margin=dict(l=40, r=20, t=80, b=40))
    fig.update_xaxes(title_text="Seconds", row=3, col=1)
    fig.update_yaxes(title_text="Watts", row=1, col=1)
    fig.update_yaxes(title_text="HR (bpm)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="NP-style Eff W", row=2, col=1)
    fig.update_yaxes(title_text="nQDI", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # roast: weight-adjusted based
    st.markdown("### 🧯 Roast (weight-adjusted nQDI decides how ugly)")
    st.info(roast_msg)

    st.caption("Tips: drag to zoom, hover to inspect seconds. Best/Worst markers show where you dominated or got dominated for that duration. Post the screenshot, lose the KOM, blame your tyre pressure, and flex your ✨aërø✨ socks.")
