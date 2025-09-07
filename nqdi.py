# app.py
"""
nQDI™: Full-featured FIT-only HR-dominant QDI app
Features:
 - FIT-only uploads (power/hr/speed/cadence)
 - Remove stopped time only when speed == 0
 - HR-aware effective watts (lower HR @ same power => better)
 - Normalized-power-style math, per-second nQDI and W/kg
 - Best/worst windows for many durations
 - Optional weights (weight optional)
 - Rotating escalating unhinged roasts
 - Leaderboard save/load (CSV)
 - Export roast report (TXT)
"""
import io
import os
import math
import random
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile

# ---------------------------
# Page config & tiny theater
# ---------------------------
STORAGE_CSV = "leaderboard.csv"
st.set_page_config(page_title="nQDI™ Live — Full Deluxe", layout="wide")
st.markdown(
    "<h1>🩸 nQDI™ Live — Quad Deficit Index: Catastrophic Deluxe Edition</h1>",
    unsafe_allow_html=True,
)
st.caption("Powered by power meters, Strava dreams, KOM envy, and the scientific art of roasting. Lower HR at same power = you're actually stronger. QDI will not lie.")

# ---------------------------
# Utilities: FIT parsing + cleaning
# ---------------------------
def read_fit_bytes(bytes_io):
    """Parse FIT file bytes and return dataframe with time (timedelta seconds), power, hr, speed, cadence."""
    fitfile = FitFile(io.BytesIO(bytes_io))
    rows = []
    for msg in fitfile.get_messages("record"):
        rec = {"timestamp": None, "power": None, "hr": None, "speed": None, "cadence": None}
        for d in msg:
            if d.name == "timestamp":
                rec["timestamp"] = pd.to_datetime(d.value)
            elif d.name == "power":
                rec["power"] = d.value
            elif d.name == "heart_rate":
                rec["hr"] = d.value
            elif d.name == "speed":
                rec["speed"] = d.value  # m/s
            elif d.name == "cadence":
                rec["cadence"] = d.value
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sec","power","hr","speed","cadence"])
    # use timestamps if available, resample to 1s
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
        # fabricate sec column
        df = df.dropna(subset=["power"]).reset_index(drop=True)
        df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
        df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
        df["speed"] = pd.to_numeric(df.get("speed", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["cadence"] = pd.to_numeric(df.get("cadence", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["sec"] = np.arange(len(df))
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce").fillna(0)
    return df[["sec","power","hr","speed","cadence"]]

# ---------------------------
# Math helpers
# ---------------------------
def rolling_np_series(power_series, window=30):
    """Return per-second NP-like series: rolling mean then 4th-power rolling mean ^0.25.
       For each second, compute NP-like value centered/ending at that second using last `window` seconds."""
    s = pd.Series(power_series).astype(float)
    # rolling mean of power over 30s then compute 4th power mean of those rolling means over available windows
    # Simpler approach: compute rolling mean (30s) then take rolling fourth power & 0.25 across same window.
    # We'll implement classic: rolling_mean = mean(power over 30s), then np = (mean(rolling_mean**4))**0.25 (over all valid)
    # But for per-second: compute rolling_mean (30s) then compute rolling (30s) of rolling_mean**4 and take ^0.25.
    if len(s) == 0:
        return np.array([])
    rm = s.rolling(window=window, min_periods=1).mean()
    p4 = (rm ** 4).rolling(window=window, min_periods=1).mean()
    np_ts = np.power(p4, 0.25).to_numpy()
    # handle NaNs: replace NaN with simple moving average fallback
    np_ts = np.nan_to_num(np_ts, nan=s.fillna(0).to_numpy())
    return np_ts

def scalar_np(power_arr, window=30):
    """Scalar NP value for entire ride (classic)."""
    s = pd.Series(power_arr).astype(float)
    if len(s) == 0:
        return 0.0
    if len(s) < window:
        return float(s.mean())
    rm = s.rolling(window=window, min_periods=window).mean().dropna()
    if rm.empty:
        return float(s.mean())
    return float(np.mean((rm ** 4)) ** 0.25)

def effective_watts_hr_dominant(power_arr, hr_arr, lthr, alpha=1.5, clip_low=0.3, clip_high=3.0):
    """Lower HR at same power => better. Uses (LTHR/HR)^alpha to scale power.
       If hr_arr has zeros (missing), scaling will treat those points as 'no HR' and return raw power for them (handled upstream)."""
    power = np.array(power_arr, dtype=float)
    hr = np.array(hr_arr, dtype=float)
    if lthr is None or lthr <= 0 or np.all(hr == 0):
        return power
    # replace 0 with nan so we don't divide by 0; we'll nan_to_num later to 1.0 (no change)
    hr_safe = np.where(hr <= 0, np.nan, hr)
    ratio = np.divide(lthr, hr_safe, out=np.full_like(hr_safe, np.nan), where=(~np.isnan(hr_safe)))
    ratio = np.clip(ratio, clip_low, clip_high)
    scale = np.nan_to_num(ratio, nan=1.0) ** alpha
    return power * scale

def compute_best_worst_windows(series_arr, durations_secs):
    """Return for each duration the best (min nQDI -> you dominated) and worst (max nQDI -> you got dominated).
       For nQDI series (friend/you), best = min, worst = max; returns values and window end indices."""
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
        best_idx = int(np.argmin(roll) + (d - 1))  # end idx
        worst_idx = int(np.argmax(roll) + (d - 1))
        out.append({"duration": d, "best_val": best_val, "best_idx": best_idx, "worst_val": worst_val, "worst_idx": worst_idx})
    return out

# ---------------------------
# Roast engine (lots of roasts)
# ---------------------------
ROAST_POOLS = {
    "elite": [
        "🚀 Overlord status: your friend is now legally a walking watt factory. Your reveal party cancelled.",
        "✨ Absolute dominance: your friend bequeaths you a water bottle and a pamphlet on humility."
    ],
    "light": [
        "⚡ Close, but your ego trips on shoelaces. Post it while it's still true.",
        "😅 Neck-and-neck — you both deserve pastries, not praise."
    ],
    "meh": [
        "💥 Mild humiliation: your watts are shaky but alive. Think intervals, not excuses.",
        "🤡 Your friend flexed; you fidgeted. Next time, try two gels."
    ],
    "bad": [
        "🔥 Leaky tire apocalypse: your pedals betrayed you mid-commitment.",
        "📉 Your FTP graph looks suspiciously like a used tissue."
    ],
    "nuclear": [
        "💣 Quad Bankruptcy: retirement, crochet, and long apologies to your drivetrain recommended.",
        "⚰️ Officially cremated: your Strava profile now a cautionary tale."
    ],
}

MICRO_ROASTS = [
    "That sprint at t=42s? Spaghetti limbs detonated.",
    "HR curve reads like a horror short film; watts sat in the corner and sobbed.",
    "Peak humiliation archived for posterity — PNG worthy.",
    "Your cadence looked like someone learning drums badly.",
    "Strava auto-pause judged you and muted notifications."
]

def pick_roast(nqdi_kg_mean):
    """Escalate roast pool by nqdi_kg_mean and pick random micro roast addition."""
    if nqdi_kg_mean is None or math.isnan(nqdi_kg_mean):
        core = "nQDI undefined — likely because someone uploaded a coffee break instead of a ride."
    elif nqdi_kg_mean < 0.9:
        core = random.choice(ROAST_POOLS["elite"])
    elif nqdi_kg_mean < 1.0:
        core = random.choice(ROAST_POOLS["light"])
    elif nqdi_kg_mean < 1.2:
        core = random.choice(ROAST_POOLS["meh"])
    elif nqdi_kg_mean < 1.6:
        core = random.choice(ROAST_POOLS["bad"])
    else:
        core = random.choice(ROAST_POOLS["nuclear"])
    micro = random.choice(MICRO_ROASTS)
    return f"{core}  \n{micro}"

# ---------------------------
# Leaderboard helpers
# ---------------------------
def save_result_to_leaderboard(entry, csv_path=STORAGE_CSV):
    """Append entry dict to CSV; create file if missing."""
    df_new = pd.DataFrame([entry])
    if os.path.exists(csv_path):
        try:
            df_exist = pd.read_csv(csv_path)
            df_all = pd.concat([df_exist, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)

def load_leaderboard(csv_path=STORAGE_CSV):
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ---------------------------
# UI: Sidebar / settings
# ---------------------------
st.sidebar.header("Settings & Options")
alpha = st.sidebar.slider("HR exponent α (higher = HR matters more)", 0.5, 2.5, 1.5, 0.1)
smooth_span = st.sidebar.slider("Smoothing (EMA span in seconds)", 1, 60, 20, 1)
np_window = st.sidebar.slider("NP rolling window (sec) — surge sensitivity", 10, 60, 30, 1)
durations_default = [15, 30, 60, 300, 600, 1200]
durations_secs = st.sidebar.multiselect("Best/Worst durations (seconds)", options=durations_default, default=[30,60,300], help="Choose durations to compute best/worst windows")
save_leaderboard_toggle = st.sidebar.checkbox("Save results to local leaderboard", value=False)
random_title = random.choice([
    "nQDI™ Live — Roast Machine of Friends",
    "nQDI™: Watt-Truth & Quad-Justice",
    "nQDI™ Deluxe — Where KOMs Get Meaner"
])
st.sidebar.markdown(f"**App alias:** {random_title}")

# ---------------------------
# UI: Main layout: tabs
# ---------------------------
tabs = st.tabs(["Compare Rides", "Leaderboard", "Settings & About"])
tab_cmp, tab_lb, tab_cfg = tabs

with tab_cmp:
    st.header("Compare two .FIT rides (FIT only, duh)")
    cols = st.columns(2)
    with cols[0]:
        you_upload = st.file_uploader("Upload YOUR .fit", type=["fit"], key="you_upload")
        you_weight = st.number_input("Your weight (kg) — optional", min_value=0.0, value=0.0, step=0.1, key="you_weight")
        you_lthr = st.number_input("Your LTHR (bpm, optional)", min_value=0, value=0, step=1, key="you_lthr")
    with cols[1]:
        friend_upload = st.file_uploader("Upload FRIEND .fit", type=["fit"], key="friend_upload")
        friend_weight = st.number_input("Friend weight (kg) — optional", min_value=0.0, value=0.0, step=0.1, key="friend_weight")
        friend_lthr = st.number_input("Friend LTHR (bpm, optional)", min_value=0, value=0, step=1, key="friend_lthr")

    run = st.button("⚡ Calculate nQDI™ (and roast politely)")

    if run:
        if (you_upload is None) or (friend_upload is None):
            st.error("Upload both .fit files — I'm not a miracle worker.")
        else:
            try:
                df_you = read_fit_bytes(you_upload.read())
                df_friend = read_fit_bytes(friend_upload.read())
            except Exception as e:
                st.exception(f"Failed to parse FIT: {e}")
                st.stop()

            # Check we have power data
            if df_you.empty or df_friend.empty:
                st.error("One of the files had no usable power data after parsing. Are you sure these are .fit rides with power?")
                st.stop()

            # Warning for HR missing
            hr_present_you = (df_you["hr"] > 0).any()
            hr_present_friend = (df_friend["hr"] > 0).any()
            if not hr_present_you or not hr_present_friend:
                st.warning("⚠️ HR missing for one or both riders — sections without HR will be computed from power alone. Accuracy reduced, humiliation preserved.")

            # Only remove stopped samples where speed == 0 (we DID that in parsing). If both have >0 length proceed.
            if len(df_you) == 0 or len(df_friend) == 0:
                st.error("No moving data after removing stops (speed==0). One of you might have uploaded a paused ride.")
                st.stop()

            # Align lengths: trim to shorter length (simple approach)
            min_len = min(len(df_you), len(df_friend))
            df_you = df_you.iloc[:min_len].reset_index(drop=True)
            df_friend = df_friend.iloc[:min_len].reset_index(drop=True)

            # Effective watts using HR-dominant rule; if LTHR missing treat those HR points as absent
            you_lthr_val = you_lthr if you_lthr > 0 else None
            friend_lthr_val = friend_lthr if friend_lthr > 0 else None
            y_eff = effective_watts_hr_dominant(df_you["power"].to_numpy(), df_you["hr"].to_numpy(), you_lthr_val, alpha=alpha)
            f_eff = effective_watts_hr_dominant(df_friend["power"].to_numpy(), df_friend["hr"].to_numpy(), friend_lthr_val, alpha=alpha)

            # Smooth (EMA) to remove jitter
            def ema_arr(x, span):
                return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()
            y_eff_s = ema_arr(y_eff, smooth_span)
            f_eff_s = ema_arr(f_eff, smooth_span)

            # Per-second NP-like series (on effective watts)
            y_np_ts = rolling_np_series(y_eff_s, window=np_window)
            f_np_ts = rolling_np_series(f_eff_s, window=np_window)

            eps = 1e-6
            nqdi_sec_raw = np.divide(f_np_ts, np.maximum(y_np_ts, eps))
            nqdi_sec_raw = np.clip(nqdi_sec_raw, 0.0, 10.0)

            # Weight-adjusted variant (if weights present)
            if you_weight > 0 and friend_weight > 0:
                y_np_wkg = y_np_ts / max(you_weight, 0.1)
                f_np_wkg = f_np_ts / max(friend_weight, 0.1)
                nqdi_sec_kg = np.divide(f_np_wkg, np.maximum(y_np_wkg, eps))
                nqdi_sec_kg = np.clip(nqdi_sec_kg, 0.0, 10.0)
            else:
                nqdi_sec_kg = nqdi_sec_raw  # fallback when weight missing

            # Summaries
            mean_raw = float(np.mean(nqdi_sec_raw))
            mean_kg = float(np.mean(nqdi_sec_kg))
            p95 = float(np.percentile(nqdi_sec_raw, 95))

            # Scalar NP shorthand
            you_np_scalar = scalar_np(y_eff_s, window=np_window)
            friend_np_scalar = scalar_np(f_eff_s, window=np_window)
            you_avg_power = float(np.mean(df_you["power"]))
            friend_avg_power = float(np.mean(df_friend["power"]))
            you_avg_hr = float(np.mean(df_you["hr"])) if hr_present_you else None
            friend_avg_hr = float(np.mean(df_friend["hr"])) if hr_present_friend else None

            # Best/worst windows
            bw = compute_best_worst_windows(nqdi_sec_kg, durations_secs if durations_secs else [30,60,300])

            # Best efforts (power) for typical durations from raw effective watts (not weight adjusted)
            bests_you = {}
            bests_friend = {}
            for d in [5,10,15,30,60,120,300,600,1200]:
                if len(y_eff_s) >= d:
                    bests_you[f"{d}s"] = float(pd.Series(y_eff_s).rolling(window=d).mean().max())
                else:
                    bests_you[f"{d}s"] = np.nan
                if len(f_eff_s) >= d:
                    bests_friend[f"{d}s"] = float(pd.Series(f_eff_s).rolling(window=d).mean().max())
                else:
                    bests_friend[f"{d}s"] = np.nan

            # Display metrics
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("💀 mean nQDI (raw)", f"{mean_raw:.3f}")
            m2.metric("⚖️ mean nQDI (W/kg)", f"{mean_kg:.3f}")
            m3.metric("🔪 nQDI peak (95%)", f"{p95:.3f}")
            m4.metric("🧮 α (HR weight)", f"{alpha:.2f}")

            st.markdown("### Rider summaries (nerdy details so you can cry privately)")
            left, right = st.columns(2)
            with left:
                st.write("**You**")
                st.write(f"- Avg power: {you_avg_power:.1f} W")
                st.write(f"- NP-like (eff): {you_np_scalar:.1f} W")
                if you_avg_hr and not math.isnan(you_avg_hr):
                    st.write(f"- Avg HR: {you_avg_hr:.0f} bpm")
                if you_weight > 0:
                    st.write(f"- Weight: {you_weight:.1f} kg -> {you_np_scalar/you_weight:.2f} W/kg (eff)")
            with right:
                st.write("**Friend**")
                st.write(f"- Avg power: {friend_avg_power:.1f} W")
                st.write(f"- NP-like (eff): {friend_np_scalar:.1f} W")
                if friend_avg_hr and not math.isnan(friend_avg_hr):
                    st.write(f"- Avg HR: {friend_avg_hr:.0f} bpm")
                if friend_weight > 0:
                    st.write(f"- Weight: {friend_weight:.1f} kg -> {friend_np_scalar/friend_weight:.2f} W/kg (eff)")

            # Best efforts table
            st.markdown("### 🔥 Best efforts (effective watts)")
            be_df = pd.DataFrame({
                "You": bests_you,
                "Friend": bests_friend
            })
            st.table(be_df.T)

            # Pick roast & show
            roast_msg = pick_roast(mean_kg)
            st.markdown("### 🧯 Roasts (rotating, escalating)")
            st.info(roast_msg)

            # Offer to save to leaderboard
            if save_leaderboard_toggle:
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "you_file": getattr(you_upload, "name", "you.fit"),
                    "friend_file": getattr(friend_upload, "name", "friend.fit"),
                    "mean_nqdi_raw": mean_raw,
                    "mean_nqdi_kg": mean_kg,
                    "you_np": you_np_scalar,
                    "friend_np": friend_np_scalar
                }
                save_result_to_leaderboard(entry)
                st.success("Saved result to local leaderboard.csv")

            # Create a textual roast report and allow download
            report_lines = []
            report_lines.append("nQDI™ Roast Report")
            report_lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC")
            report_lines.append(f"You file: {getattr(you_upload,'name','you.fit')}")
            report_lines.append(f"Friend file: {getattr(friend_upload,'name','friend.fit')}")
            report_lines.append("")
            report_lines.append(f"Mean nQDI (raw): {mean_raw:.3f}")
            report_lines.append(f"Mean nQDI (W/kg): {mean_kg:.3f}")
            report_lines.append(f"Peak nQDI (95%): {p95:.3f}")
            report_lines.append("")
            report_lines.append("You — stats:")
            report_lines.append(f"  Avg power: {you_avg_power:.1f} W")
            report_lines.append(f"  NP-like (eff): {you_np_scalar:.1f} W")
            if you_avg_hr and not math.isnan(you_avg_hr):
                report_lines.append(f"  Avg HR: {you_avg_hr:.0f} bpm")
            if you_weight > 0:
                report_lines.append(f"  Weight: {you_weight:.1f} kg")
            report_lines.append("")
            report_lines.append("Friend — stats:")
            report_lines.append(f"  Avg power: {friend_avg_power:.1f} W")
            report_lines.append(f"  NP-like (eff): {friend_np_scalar:.1f} W")
            if friend_avg_hr and not math.isnan(friend_avg_hr):
                report_lines.append(f"  Avg HR: {friend_avg_hr:.0f} bpm")
            if friend_weight > 0:
                report_lines.append(f"  Weight: {friend_weight:.1f} kg")
            report_lines.append("")
            report_lines.append("Best efforts (You):")
            for k,v in bests_you.items():
                report_lines.append(f"  {k}: {v:.1f} W")
            report_lines.append("Best efforts (Friend):")
            for k,v in bests_friend.items():
                report_lines.append(f"  {k}: {v:.1f} W")
            report_lines.append("")
            report_lines.append("Roast:")
            report_lines.append(roast_msg)
            report_txt = "\n".join(report_lines)
            st.download_button("Download roast report (txt)", data=report_txt, file_name="nqdi_roast_report.txt", mime="text/plain")

            # Plot
            st.markdown("### 📈 Plots (hover & zoom to laugh/cry)")
            t = np.arange(len(nqdi_sec_raw))
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                specs=[[{"secondary_y": True}], [{}], [{}]],
                                subplot_titles=("Power & HR", "Effective NP-style watts (smoothed)", "nQDI (raw & W/kg)"))

            # Top: power and HR
            fig.add_trace(go.Scatter(x=t, y=df_you["power"], name="You Power (W)"), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=df_friend["power"], name="Friend Power (W)"), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=df_you["hr"], name="You HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(x=t, y=df_friend["hr"], name="Friend HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Power (W)", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="HR (bpm)", row=1, col=1, secondary_y=True)

            # Middle: NP-like effective watts
            fig.add_trace(go.Scatter(x=t, y=y_np_ts, name="You NP-like (eff W)"), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=f_np_ts, name="Friend NP-like (eff W)"), row=2, col=1)
            fig.update_yaxes(title_text="NP-like Eff W", row=2, col=1)

            # Bottom: nQDI
            fig.add_trace(go.Scatter(x=t, y=nqdi_sec_raw, name="nQDI Raw"), row=3, col=1)
            fig.add_trace(go.Scatter(x=t, y=nqdi_sec_kg, name="nQDI W/kg"), row=3, col=1)
            fig.add_hline(y=1.0, line_dash="dash", annotation_text="Break-even", row=3, col=1)
            fig.update_yaxes(title_text="nQDI", row=3, col=1)

            fig.update_layout(height=900, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)

with tab_lb:
    st.header("Leaderboard (local CSV)")
    lb = load_leaderboard()
    if lb.empty:
        st.info("No saved leaderboard entries yet. Enable saving in Settings & run a comparison to populate this.")
    else:
        st.dataframe(lb.sort_values("timestamp", ascending=False).reset_index(drop=True))
        if st.button("Clear leaderboard (delete local CSV)"):
            try:
                os.remove(STORAGE_CSV)
                st.success("Leaderboard cleared.")
            except Exception as e:
                st.error(f"Could not delete leaderboard: {e}")

with tab_cfg:
    st.header("Settings & About")
    st.markdown("""
    **What this app does (short):**  
    - Reads `.fit` files (power, HR, speed, cadence) and removes stopped time (speed==0).  
    - Computes HR-dominant effective watts: lower HR at the same power increases effective watts (you're efficient).  
    - Uses Normalized-Power-style math so surges matter.  
    - Produces per-second nQDI (friend / you) and weight-adjusted nQDI if weights are provided.  
    - Best/worst windows, best efforts, rotating escalating roasts.  

    **Privacy / Storage:**  
    - Files are processed in-memory. If you enable saving to the leaderboard, summary rows are appended to `leaderboard.csv` in the app directory. If you deploy publicly, review data storage and consent rules (don’t be a jerk).  

    **Dependencies**: `streamlit pandas numpy plotly fitparse`
    """)
    st.markdown("### Sample unhinged tagline")
    st.write(random.choice([
        "nQDI™ — Because your friend deserves public humiliation in high-definition.",
        "nQDI™ — Turning watts into tears since you installed a power meter.",
        "nQDI™ — For when a KOM is a personality trait, not a metric."
    ]))

# ---------------------------
# End
# ---------------------------
st.caption("Tip: want CSV exports or Strava automation? Tell me and I'll happily make you morally ambiguous. And yes — mention your ✨aërø✨ socks in the group chat for full effect.")
