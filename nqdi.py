# app.py
"""
nQDI™ — Polished FIT-only HR-dominant QDI app
Drop into a repo and run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io
import os
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from fitparse import FitFile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Config & small theatre
# -------------------------
LEADERBOARD_CSV = "leaderboard.csv"
APP_TITLE = "nQDI™ Live — Quad Deficit Index: Polished Roast Edition"
# fun rotating subtitle (unhinged but not obscene)
SUBTITLES = [
    "Watt truth, HR honesty, KOM humiliation.",
    "Because your friend deserves public shame in high-definition.",
    "Lower HR at same watts = cooler legs. QDI won't lie.",
]
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"# 🩸 {APP_TITLE}")
st.caption(f"{random.choice(SUBTITLES)} • powered by power meters, Strava data vibes, KOM envy, and the sacred Quad Deficit Index (QDI). Also flex your ✨aërø✨ socks and çårbøn fïbër fantasies.")

# -------------------------
# Utility: parse FIT -> DataFrame
# -------------------------
def read_fit_bytes(bytes_io):
    """Parse FIT bytes and return DataFrame with columns: sec,power,hr,speed,cadence."""
    fit = FitFile(io.BytesIO(bytes_io))
    rows = []
    for msg in fit.get_messages("record"):
        rec = {"timestamp": None, "power": None, "hr": None, "speed": None, "cadence": None}
        for d in msg:
            if d.name == "timestamp":
                rec["timestamp"] = pd.to_datetime(d.value)
            elif d.name == "power":
                rec["power"] = d.value
            elif d.name == "heart_rate":
                rec["hr"] = d.value
            elif d.name == "speed":
                rec["speed"] = d.value
            elif d.name == "cadence":
                rec["cadence"] = d.value
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sec","power","hr","speed","cadence"])

    # If timestamps exist -> resample to 1s
    if df["timestamp"].notna().any():
        df = df.dropna(subset=["power"]).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df = df.resample("1S").mean()
        df["hr"] = df["hr"].ffill().fillna(0)
        df["speed"] = df["speed"].ffill().fillna(0)
        df["cadence"] = df["cadence"].ffill().fillna(0)
        df = df.reset_index()
        start = df["timestamp"].iloc[0]
        df["sec"] = (df["timestamp"] - start).dt.total_seconds().astype(int)
    else:
        df = df.dropna(subset=["power"]).reset_index(drop=True)
        df["sec"] = np.arange(len(df))

    # ensure numeric & fill
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["speed"] = pd.to_numeric(df.get("speed", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    df["cadence"] = pd.to_numeric(df.get("cadence", pd.Series(0, index=df.index)), errors="coerce").fillna(0)

    return df[["sec","power","hr","speed","cadence"]]

# -------------------------
# Math helpers
# -------------------------
def ema_arr(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()

def rolling_np_series(power_series, window=30):
    """Per-second NP-style series computed on input power array.
       Implementation: rm = rolling_mean(power, window), p4 = rolling_mean(rm**4, window), np_ts = p4**0.25
    """
    s = pd.Series(power_series).astype(float)
    if len(s) == 0:
        return np.array([])
    rm = s.rolling(window=window, min_periods=1).mean()
    p4 = (rm ** 4).rolling(window=window, min_periods=1).mean()
    np_ts = np.power(p4.to_numpy(), 0.25)
    # fallback: replace NaN with simple moving mean
    np_ts = np.nan_to_num(np_ts, nan=s.rolling(window=window, min_periods=1).mean().to_numpy())
    return np_ts

def scalar_np(power_arr, window=30):
    s = pd.Series(power_arr).astype(float)
    if len(s) == 0: return 0.0
    if len(s) < window:
        return float(s.mean())
    rm = s.rolling(window=window, min_periods=window).mean().dropna()
    if rm.empty:
        return float(s.mean())
    return float(np.mean(rm ** 4) ** 0.25)

def effective_watts_hr_dominant(power_arr, hr_arr, lthr, alpha=1.5, clip_low=0.3, clip_high=3.0):
    """Scale power by (LTHR / HR)^alpha so lower HR at same power increases effective watts.
       If lthr missing or hr array entirely zero -> return raw power array (no HR scaling).
    """
    power = np.array(power_arr, dtype=float)
    hr = np.array(hr_arr, dtype=float)
    if lthr is None or lthr <= 0 or np.all(hr == 0):
        return power
    # avoid div-by-zero: mark missing hr as NaN (no scaling)
    hr_safe = np.where(hr <= 0, np.nan, hr)
    ratio = np.divide(lthr, hr_safe, out=np.full_like(hr_safe, np.nan), where=~np.isnan(hr_safe))
    ratio = np.clip(ratio, clip_low, clip_high)
    scale = np.nan_to_num(ratio, nan=1.0) ** alpha
    return power * scale

def compute_best_efforts(arr, windows_s):
    """Return dict of max-average power for each window in windows_s."""
    out = {}
    a = np.array(arr, dtype=float)
    N = len(a)
    for d in windows_s:
        if N >= d and d > 0:
            roll = pd.Series(a).rolling(window=d).mean().dropna()
            out[d] = float(roll.max()) if not roll.empty else float("nan")
        else:
            out[d] = float("nan")
    return out

def compute_best_worst_windows(nqdi_arr, durations_secs):
    """For durations (seconds) compute best (min) and worst (max) rolling means and their end indices."""
    return compute_best_worst_windows.__wrapped__(nqdi_arr, durations_secs) if hasattr(compute_best_worst_windows, "__wrapped__") else _local_best_worst(nqdi_arr, durations_secs)

def _local_best_worst(series_arr, durations_secs):
    arr = np.array(series_arr, dtype=float)
    N = len(arr)
    out = []
    for d in durations_secs:
        if d <= 0 or N < d:
            out.append({"duration": d, "best_val": None, "best_idx": None, "worst_val": None, "worst_idx": None})
            continue
        roll = pd.Series(arr).rolling(window=d, min_periods=d).mean().dropna().to_numpy()
        if roll.size == 0:
            out.append({"duration": d, "best_val": None, "best_idx": None, "worst_val": None, "worst_idx": None})
            continue
        best_val = float(np.min(roll))
        worst_val = float(np.max(roll))
        best_idx = int(np.argmin(roll) + (d - 1))
        worst_idx = int(np.argmax(roll) + (d - 1))
        out.append({"duration": d, "best_val": best_val, "best_idx": best_idx, "worst_val": worst_val, "worst_idx": worst_idx})
    return out

# -------------------------
# Roast engine (polished & huge)
# -------------------------
ROAST_POOLS = {
    "elite": [
        "🚀 Elite menace: your friend is actually producing existential watts. Consider retiring politely.",
        "✨ Overlord: your friend just filed a restraining order against smaller power numbers."
    ],
    "light": [
        "⚡ Close race — post it to Strava before reality notices.",
        "😏 Neck-and-neck; bring pastries, not excuses."
    ],
    "meh": [
        "💥 Mild humiliation: some effort, much weeping.",
        "🤡 You tried. The watts tried harder."
    ],
    "bad": [
        "🔥 Pain cave alert: your quads requested early retirement papers.",
        "📉 Your FTP graph looks like a sad tweet thread."
    ],
    "nuclear": [
        "💣 Quad Bankruptcy: donate your bike and take up knitting.",
        "⚰️ Catastrophic: your Strava profile is now a cautionary tale."
    ],
}

MICRO_ROASTS = [
    "That sprint at t=42s? Spaghetti limbs detonated.",
    "HR curve reads like a horror short; watts sobbed in the corner.",
    "Peak nQDI archived for posterity — PNG worthy.",
    "Cadence looked like someone learning drums badly.",
    "Strava auto-pause judged you and muted notifications.",
    "Your ✨aërø✨ socks win style awards; your legs do not."
]

def pick_roast(mean_nqdi_kg):
    if mean_nqdi_kg is None or math.isnan(mean_nqdi_kg):
        core = "nQDI undefined — you uploaded a coffee break instead of a ride."
    elif mean_nqdi_kg < 0.9:
        core = random.choice(ROAST_POOLS["elite"])
    elif mean_nqdi_kg < 1.0:
        core = random.choice(ROAST_POOLS["light"])
    elif mean_nqdi_kg < 1.2:
        core = random.choice(ROAST_POOLS["meh"])
    elif mean_nqdi_kg < 1.6:
        core = random.choice(ROAST_POOLS["bad"])
    else:
        core = random.choice(ROAST_POOLS["nuclear"])
    micro = random.choice(MICRO_ROASTS)
    return f"{core}  \n{micro}"

# -------------------------
# Leaderboard helpers
# -------------------------
def save_leaderboard_row(row, csv_path=LEADERBOARD_CSV):
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        try:
            df_exists = pd.read_csv(csv_path)
            df_out = pd.concat([df_exists, df_new], ignore_index=True)
        except Exception:
            df_out = df_new
    else:
        df_out = df_new
    df_out.to_csv(csv_path, index=False)

def load_leaderboard(csv_path=LEADERBOARD_CSV):
    if os.path.exists(csv_path):
        try: return pd.read_csv(csv_path)
        except Exception: return pd.DataFrame()
    return pd.DataFrame()

# -------------------------
# Sidebar controls (polish)
# -------------------------
st.sidebar.header("nQDI™ Settings")
alpha = st.sidebar.slider("HR exponent α (how much HR shifts effective watts)", 0.5, 2.5, 1.5, 0.1)
smooth_span = st.sidebar.slider("Smoothing (EMA span seconds)", 1, 60, 20, 1)
np_window = st.sidebar.slider("NP window (sec) for surge sensitivity", 10, 60, 30, 1)
durations_default = [15,30,60,300,600,1200]
durations_secs = st.sidebar.multiselect("Best/worst windows (sec)", options=durations_default, default=[30,60,300])
save_toggle = st.sidebar.checkbox("Save summary to local leaderboard (leaderboard.csv)", value=False)
demo_mode = st.sidebar.checkbox("Demo mode (generate fake rides for quick screenshot)", value=False)

st.sidebar.markdown("Privacy: files are processed in-memory. If you enable leaderboard saving, a small summary row is stored locally.")

# -------------------------
# Main: tabs
# -------------------------
tabs = st.tabs(["Compare Rides", "Leaderboard", "About & Launch Checklist"])
tab_cmp, tab_lb, tab_about = tabs

with tab_cmp:
    st.header("Compare two .FIT rides")
    cols = st.columns(2)

    with cols[0]:
        you_file = st.file_uploader("Upload YOUR .fit", type=["fit"], key="you")
        you_name = you_file.name if you_file is not None else None
        you_weight = st.number_input("Your weight (kg, optional)", min_value=0.0, value=0.0, step=0.1, key="you_weight")
        you_lthr = st.number_input("Your LTHR (bpm, optional)", min_value=0, value=0, step=1, key="you_lthr")

    with cols[1]:
        friend_file = st.file_uploader("Upload FRIEND .fit", type=["fit"], key="friend")
        friend_name = friend_file.name if friend_file is not None else None
        friend_weight = st.number_input("Friend weight (kg, optional)", min_value=0.0, value=0.0, step=0.1, key="friend_weight")
        friend_lthr = st.number_input("Friend LTHR (bpm, optional)", min_value=0, value=0, step=1, key="friend_lthr")

    if demo_mode:
        st.info("Demo mode ON — generating two synthetic rides for screenshot/demo. No uploads required.")
        # simple synthetic rides: sinusoidal surges, friend is stronger
        t = np.arange(0, 1800)
        you_df = pd.DataFrame({
            "sec": t,
            "power": 180 + 40*np.sin(t/10) + (np.random.randn(len(t))*6),
            "hr": 145 + 6*np.sin(t/20) + (np.random.randn(len(t))*2),
            "speed": 7 + np.abs(0.2*np.sin(t/15)),
            "cadence": 80 + 8*np.sin(t/12)
        }).astype(float)
        friend_df = pd.DataFrame({
            "sec": t,
            "power": 200 + 55*np.sin(t/9) + (np.random.randn(len(t))*6),
            "hr": 138 + 5*np.sin(t/21) + (np.random.randn(len(t))*2),
            "speed": 7.5 + np.abs(0.2*np.sin(t/13)),
            "cadence": 85 + 7*np.sin(t/11)
        }).astype(float)
        you_filename = "demo_you.fit"
        friend_filename = "demo_friend.fit"
    else:
        you_df = None
        friend_df = None
        you_filename = you_name
        friend_filename = friend_name

    compute_btn = st.button("⚡ Compute nQDI™ (generate roast)")

    if compute_btn:
        # load files or use demo
        if demo_mode:
            df_you = you_df
            df_friend = friend_df
        else:
            if (you_file is None) or (friend_file is None):
                st.error("Upload both .fit files (you + friend) or enable Demo mode.")
                st.stop()
            try:
                df_you = read_fit_bytes(you_file.read())
                df_friend = read_fit_bytes(friend_file.read())
                you_filename = getattr(you_file, "name", "you.fit")
                friend_filename = getattr(friend_file, "name", "friend.fit")
            except Exception as e:
                st.exception(f"Failed to parse FIT files: {e}")
                st.stop()

        # sanity
        if df_you.empty or df_friend.empty:
            st.error("One ride had no usable power data after parsing.")
            st.stop()

        # At parse time we dropped speed==0 samples already; if that removed everything warn
        if len(df_you) == 0 or len(df_friend) == 0:
            st.error("No moving data after removing stops (speed==0). Check your FIT files.")
            st.stop()

        # align by sec: inner merge on sec to keep only overlapping seconds
        merged = pd.merge(df_you, df_friend, on="sec", suffixes=("_you","_friend"))
        if merged.empty:
            # fallback: trim to min length
            min_len = min(len(df_you), len(df_friend))
            merged = pd.concat([
                df_you.iloc[:min_len].reset_index(drop=True),
                df_friend.iloc[:min_len].reset_index(drop=True)
            ], axis=1)
            merged.columns = [c+"_you" if not c.endswith("_friend") else c for c in merged.columns]

        # detect HR presence
        hr_you_present = (merged["hr_you"] > 0).any()
        hr_friend_present = (merged["hr_friend"] > 0).any()
        if not hr_you_present or not hr_friend_present:
            st.warning("⚠️ HR missing for one or both riders — sections without HR will be treated using raw power (no HR scaling). Accuracy reduced but roast intact.")

        # compute effective watts
        you_lthr_val = you_lthr if you_lthr > 0 else None
        friend_lthr_val = friend_lthr if friend_lthr > 0 else None
        y_eff = effective_watts_hr_dominant(merged["power_you"].to_numpy(), merged["hr_you"].to_numpy(), you_lthr_val, alpha=alpha)
        f_eff = effective_watts_hr_dominant(merged["power_friend"].to_numpy(), merged["hr_friend"].to_numpy(), friend_lthr_val, alpha=alpha)

        # smoothing
        y_eff_s = ema_arr(y_eff, smooth_span)
        f_eff_s = ema_arr(f_eff, smooth_span)

        # per-second NP-style series
        y_np_ts = rolling_np_series(y_eff_s, window=np_window)
        f_np_ts = rolling_np_series(f_eff_s, window=np_window)

        # per-second nQDI
        eps = 1e-6
        nqdi_raw = np.divide(f_np_ts, np.maximum(y_np_ts, eps))
        nqdi_raw = np.clip(nqdi_raw, 0.0, 10.0)

        # weight-adjusted
        if you_weight > 0 and friend_weight > 0:
            y_np_wkg = y_np_ts / max(you_weight, 0.1)
            f_np_wkg = f_np_ts / max(friend_weight, 0.1)
            nqdi_wkg = np.divide(f_np_wkg, np.maximum(y_np_wkg, eps))
            nqdi_wkg = np.clip(nqdi_wkg, 0.0, 10.0)
            used_wkg = True
        else:
            nqdi_wkg = nqdi_raw.copy()
            used_wkg = False

        # summaries
        mean_raw = float(np.mean(nqdi_raw)) if len(nqdi_raw)>0 else float("nan")
        mean_wkg = float(np.mean(nqdi_wkg)) if len(nqdi_wkg)>0 else float("nan")
        p95 = float(np.percentile(nqdi_raw, 95)) if len(nqdi_raw)>0 else float("nan")
        you_np_scalar = scalar_np(y_eff_s, window=np_window)
        friend_np_scalar = scalar_np(f_eff_s, window=np_window)
        you_avg_power = float(np.mean(merged["power_you"]))
        friend_avg_power = float(np.mean(merged["power_friend"]))
        you_avg_hr = float(np.mean(merged["hr_you"])) if hr_you_present else None
        friend_avg_hr = float(np.mean(merged["hr_friend"])) if hr_friend_present else None

        # best efforts (effective watts)
        windows_for_bests = [5,10,15,30,60,120,300,600,1200]
        bests_you = compute_best_efforts(y_eff_s, windows_for_bests)
        bests_friend = compute_best_efforts(f_eff_s, windows_for_bests)

        # best/worst nQDI windows for durations selected
        bw = _local_best_worst(nqdi_wkg, durations_secs if durations_secs else [30,60,300])

        # Display top-line metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("💀 mean nQDI (raw)", f"{mean_raw:.3f}")
        c2.metric("⚖️ mean nQDI (W/kg)" + (" (weights used)" if used_wkg else ""), f"{mean_wkg:.3f}")
        c3.metric("🔪 nQDI peak (95%)", f"{p95:.3f}")
        c4.metric("🧮 α (HR weight)", f"{alpha:.2f}")

        # rider boxes
        st.markdown("### Rider summaries")
        r1, r2 = st.columns(2)
        with r1:
            st.write("**You**")
            st.write(f"- File: `{you_filename}`")
            st.write(f"- Avg power: {you_avg_power:.1f} W")
            st.write(f"- NP-like (eff): {you_np_scalar:.1f} W")
            if you_avg_hr is not None: st.write(f"- Avg HR: {you_avg_hr:.0f} bpm")
            if you_weight>0: st.write(f"- Weight: {you_weight:.1f} kg → eff W/kg: {you_np_scalar/you_weight:.2f}")
        with r2:
            st.write("**Friend**")
            st.write(f"- File: `{friend_filename}`")
            st.write(f"- Avg power: {friend_avg_power:.1f} W")
            st.write(f"- NP-like (eff): {friend_np_scalar:.1f} W")
            if friend_avg_hr is not None: st.write(f"- Avg HR: {friend_avg_hr:.0f} bpm")
            if friend_weight>0: st.write(f"- Weight: {friend_weight:.1f} kg → eff W/kg: {friend_np_scalar/friend_weight:.2f}")

        # best efforts table
        st.markdown("### 🔥 Best efforts (effective watts)")
        be_df = pd.DataFrame({"You": bests_you, "Friend": bests_friend})
        st.dataframe(be_df.T.style.format("{:.1f}"))

        # roast selection
        roast_msg = pick_roast(mean_wkg)
        st.markdown("### 🧯 Roasts (rotating & escalating)")
        st.info(roast_msg)

        # save to leaderboard option
        if save_toggle:
            row = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "you_file": you_filename,
                "friend_file": friend_filename,
                "mean_nqdi_raw": mean_raw,
                "mean_nqdi_wkg": mean_wkg,
                "you_np": you_np_scalar,
                "friend_np": friend_np_scalar
            }
            save_leaderboard_row(row)
            st.success("Saved result to local leaderboard.csv")

        # downloadable roast report
        report = []
        report.append("nQDI™ Roast Report")
        report.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
        report.append(f"You file: {you_filename}")
        report.append(f"Friend file: {friend_filename}")
        report.append("")
        report.append(f"Mean nQDI (raw): {mean_raw:.3f}")
        report.append(f"Mean nQDI (W/kg): {mean_wkg:.3f}")
        report.append(f"Peak nQDI (95%): {p95:.3f}")
        report.append("")
        report.append("You — stats:")
        report.append(f"  Avg power: {you_avg_power:.1f} W")
        report.append(f"  NP-like (eff): {you_np_scalar:.1f} W")
        if you_avg_hr is not None: report.append(f"  Avg HR: {you_avg_hr:.0f} bpm")
        if you_weight>0: report.append(f"  Weight: {you_weight:.1f} kg")
        report.append("")
        report.append("Friend — stats:")
        report.append(f"  Avg power: {friend_avg_power:.1f} W")
        report.append(f"  NP-like (eff): {friend_np_scalar:.1f} W")
        if friend_avg_hr is not None: report.append(f"  Avg HR: {friend_avg_hr:.0f} bpm")
        if friend_weight>0: report.append(f"  Weight: {friend_weight:.1f} kg")
        report.append("")
        report.append("Best efforts (You):")
        for k,v in bests_you.items(): report.append(f"  {k}s: {v:.1f} W")
        report.append("Best efforts (Friend):")
        for k,v in bests_friend.items(): report.append(f"  {k}s: {v:.1f} W")
        report.append("")
        report.append("Roast:")
        report.append(roast_msg)
        report_txt = "\n".join(report)
        st.download_button("Download roast report (TXT)", data=report_txt, file_name="nqdi_roast_report.txt", mime="text/plain")

        # plots
        st.markdown("### 📈 Interactive Plots (zoom, hover, export PNG via camera icon)")
        t_idx = np.arange(len(nqdi_raw))
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            specs=[[{"secondary_y": True}], [{}], [{}]],
                            subplot_titles=("Power & HR", "NP-style effective watts", "nQDI (raw & W/kg)"))

        # top traces
        fig.add_trace(go.Scatter(x=t_idx, y=merged["power_you"], name="You Power (W)"), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=t_idx, y=merged["power_friend"], name="Friend Power (W)"), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=t_idx, y=merged["hr_you"], name="You HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=t_idx, y=merged["hr_friend"], name="Friend HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Power (W)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="HR (bpm)", row=1, col=1, secondary_y=True)

        # mid traces
        fig.add_trace(go.Scatter(x=t_idx, y=y_np_ts, name="You NP-like (eff W)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_idx, y=f_np_ts, name="Friend NP-like (eff W)"), row=2, col=1)
        fig.update_yaxes(title_text="NP-like Eff W", row=2, col=1)

        # bottom nQDI
        fig.add_trace(go.Scatter(x=t_idx, y=nqdi_raw, name="nQDI Raw"), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_idx, y=nqdi_wkg, name="nQDI W/kg"), row=3, col=1)
        fig.add_hline(y=1.0, line_dash="dash", annotation_text="Break-even", row=3, col=1)
        fig.update_yaxes(title_text="nQDI", row=3, col=1)

        fig.update_layout(height=900, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)


with tab_lb:
    st.header("Leaderboard (local)")
    lb = load_leaderboard()
    if lb.empty:
        st.info("No saved entries yet. Toggle 'Save to leaderboard' in the sidebar and run a comparison to populate.")
    else:
        st.dataframe(lb.sort_values("timestamp_utc", ascending=False).reset_index(drop=True))
        if st.button("Clear local leaderboard"):
            try:
                os.remove(LEADERBOARD_CSV)
                st.success("Leaderboard cleared.")
            except Exception as e:
                st.error(f"Failed to clear leaderboard: {e}")

with tab_about:
    st.header("About, Privacy, & Launch Checklist")
    st.markdown("""
    **What this app does**
    - FIT-only. Reads power, HR, speed, cadence. Removes stopped time (speed==0).
    - HR-dominant effective watts: lower HR at same watts -> higher effective watts.
    - NP-style smoothing and per-second nQDI (friend / you).
    - Best efforts, best/worst windows, rotating escalating roasts, optional leaderboard.

    **Privacy**
    - Files are processed in-memory.
    - If you enable leaderboard saving, small summary rows are stored in `leaderboard.csv`.
    - Do not enable saving on a public deploy unless everyone consents.

    **Quick Launch Checklist**
    1. Create a GitHub repo and add `app.py` + `requirements.txt`.
    2. Test locally: `pip install -r requirements.txt` then `streamlit run app.py`.
    3. If deploying to Streamlit Cloud, disable leaderboard saving or implement proper storage/consent.
    4. Optional: connect Strava OAuth later (I can scaffold this).

    **Dependencies**
    - streamlit, pandas, numpy, plotly, fitparse
    """)

st.caption("Final note: this app will mercilessly roast watts, HR, KOM dreams, and fragile egos. Use responsibly. Also: mention ✨aërø✨ socks in the group chat; results may be spicier.")
