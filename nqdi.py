# app.py
"""
nQDI™ Ultimate Launch Edition — FIT-only, HR-dominant, roast-overkill
Drop-in Streamlit app with extra charts, Spite Index™, fake metrics, demo rides, and leaderboard.
Dependencies: streamlit, pandas, numpy, plotly, fitparse
"""

import io
import os
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile

# ---------- CONFIG ----------
LEADERBOARD_CSV = "nqdi_leaderboard.csv"
st.set_page_config(page_title="nQDI™ Ultimate — Roast Machine", layout="wide")
st.title("🩸 nQDI™ Ultimate — Quad Deficit Index: Launch Edition")
st.caption("Powered by power meters, Strava drama, KOM delusions, and the sacred Quad Deficit Index (QDI).")

# ---------- Helpers: FIT parsing ----------
@st.cache_data
def parse_fit_bytes(bytes_io):
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
                rec["speed"] = d.value
            elif d.name == "cadence":
                rec["cadence"] = d.value
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["sec","power","hr","speed","cadence"])
    df = df.dropna(subset=["power"]).copy()
    # If timestamps exist, resample to 1s and create sec
    if df["timestamp"].notna().any():
        df = df.set_index("timestamp").sort_index()
        df = df.resample("1S").mean()
        df["hr"] = df["hr"].ffill().fillna(0)
        df["speed"] = df["speed"].ffill().fillna(0)
        df["cadence"] = df["cadence"].ffill().fillna(0)
        df["power"] = df["power"].fillna(0)
        df = df.reset_index()
        start = df["timestamp"].iloc[0]
        df["sec"] = (df["timestamp"] - start).dt.total_seconds().astype(int)
    else:
        df = df.reset_index(drop=True)
        df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
        df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
        df["speed"] = pd.to_numeric(df.get("speed", 0), errors="coerce").fillna(0)
        df["cadence"] = pd.to_numeric(df.get("cadence", 0), errors="coerce").fillna(0)
        df["sec"] = np.arange(len(df))
    # filter stopped: speed <= 0 considered stopped; keep only moving
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    df = df[df["speed"] > 0].reset_index(drop=True)
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce").fillna(0)
    return df[["sec","power","hr","speed","cadence"]]

# ---------- Math helpers ----------
def scalar_normalized_power(power_arr, window=30):
    s = pd.Series(power_arr).astype(float)
    if len(s) == 0:
        return 0.0
    if len(s) < window:
        return float(s.mean())
    rm = s.rolling(window=window, min_periods=window).mean().dropna()
    if rm.empty:
        return float(s.mean())
    return float(np.mean(rm**4) ** 0.25)

def ts_normalized_power(power_arr, window=30):
    s = pd.Series(power_arr).astype(float)
    if len(s) == 0:
        return np.array([])
    rm = s.rolling(window=window, min_periods=1).mean()
    p4 = (rm ** 4).rolling(window=window, min_periods=1).mean()
    np_ts = np.power(p4, 0.25).to_numpy()
    # fallback nan handling
    if np.all(np.isnan(np_ts)):
        return np.nan_to_num(np_ts, nan=np.nanmean(np_ts) if np.any(~np.isnan(np_ts)) else s.to_numpy())
    return np.nan_to_num(np_ts, nan=np.nanmean(np_ts) if np.any(~np.isnan(np_ts)) else s.to_numpy())

def effective_watts_hr(power_arr, hr_arr, lthr, alpha=1.5, clip_low=0.3, clip_high=3.0):
    power = np.array(power_arr, dtype=float)
    hr = np.array(hr_arr, dtype=float)
    if lthr is None or lthr <= 0 or np.all(hr <= 0):
        return power
    hr_safe = np.where(hr <= 0, np.nan, hr)
    ratio = np.divide(lthr, hr_safe, out=np.full_like(hr_safe, np.nan), where=~np.isnan(hr_safe))
    ratio = np.clip(ratio, clip_low, clip_high)
    scale = np.nan_to_num(ratio, nan=1.0) ** alpha
    return power * scale

def best_effort_series(power_arr, durations):
    out = {}
    s = pd.Series(power_arr).astype(float)
    for d in durations:
        if len(s) >= d and d > 0:
            out[d] = float(s.rolling(window=d).mean().max())
        else:
            out[d] = None
    return out

def compute_best_worst_windows(arr, durations_secs):
    arr_s = pd.Series(arr).astype(float)
    N = len(arr_s)
    results = []
    for d in durations_secs:
        if d <= 0 or N < d:
            results.append({"duration": d, "best_val": None, "best_idx": None, "worst_val": None, "worst_idx": None})
            continue
        roll = arr_s.rolling(window=d, min_periods=d).mean().dropna().to_numpy()
        if roll.size == 0:
            results.append({"duration": d, "best_val": None, "best_idx": None, "worst_val": None, "worst_idx": None})
            continue
        best_val = float(np.min(roll))
        worst_val = float(np.max(roll))
        best_idx = int(np.argmin(roll) + (d - 1))
        worst_idx = int(np.argmax(roll) + (d - 1))
        results.append({"duration": d, "best_val": best_val, "best_idx": best_idx, "worst_val": worst_val, "worst_idx": worst_idx})
    return results

# ---------- Roast engine ----------
ROAST_BUCKETS = {
    "elite": [
        "🚀 Overlord: your friend now rents a tiny office for their quads; your quads are on a couch tenancy agreement.",
        "✨ Legendary: they pedal politely and still leave your ego in the gutter."
    ],
    "close": [
        "⚡ Close duel — both of you suffer, but one of you does it with dignity (not you, probably).",
        "😅 Neck and neck: bring pastries, not excuses."
    ],
    "meh": [
        "💥 Mild humiliation: watts trembled and considered early retirement.",
        "🤡 You brought intervals to a cake ride... and lost your cake."
    ],
    "bad": [
        "🔥 Leaky tire apocalypse: your pedals betrayed the team and joined the opposition.",
        "📉 Your FTP chart now doubles as modern abstract art."
    ],
    "nuclear": [
        "💣 Quad Bankruptcy: file for emotional bankruptcy and consider knitting.",
        "⚰️ Extinction-level humiliation: your Strava feed will need a therapist."
    ]
}
MICRO_ROASTS = [
    "That sprint at t=42s? Spaghetti limbs exploded.",
    "HR curve: horror short film. Watts hid in the bushes.",
    "Peak humiliation archived — PNG ready for meme export.",
    "Cadence resembled someone learning drums badly.",
    "Strava auto-pause whispered 'please stop'."
]

def choose_roast(mean_qdi_wkg):
    if mean_qdi_wkg is None or math.isnan(mean_qdi_wkg):
        core = "nQDI undefined — probably missing HR and weight and dignity."
    elif mean_qdi_wkg < 0.9:
        core = random.choice(ROAST_BUCKETS["elite"])
    elif mean_qdi_wkg < 1.0:
        core = random.choice(ROAST_BUCKETS["close"])
    elif mean_qdi_wkg < 1.2:
        core = random.choice(ROAST_BUCKETS["meh"])
    elif mean_qdi_wkg < 1.6:
        core = random.choice(ROAST_BUCKETS["bad"])
    else:
        core = random.choice(ROAST_BUCKETS["nuclear"])
    micro = random.choice(MICRO_ROASTS)
    # Add some decorative nonsense (makes people spit drinks)
    nonsense = random.choice([
        "Sock Height Efficiency Index: 0.42 (wear more ✨aërø✨ socks).",
        "KOMfort Ratio: your comfort is inversely proportional to friend's smugness.",
        "Quad Bankruptcy Timer: 00:03:12 (approximate existential collapse)."
    ])
    return f"{core}  \n{micro}  \n{nonsense}"

# ---------- Leaderboard ----------
def save_entry(entry, csv_path=LEADERBOARD_CSV, anonymize=True):
    df_new = pd.DataFrame([entry])
    if anonymize:
        # mask filenames
        df_new["you_file"] = "you.fit"
        df_new["friend_file"] = "friend.fit"
    if os.path.exists(csv_path):
        try:
            df_exist = pd.read_csv(csv_path)
            df_all = pd.concat([df_exist, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)

def load_leaderboard(csv_path=LEADERBOARD_CSV):
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ---------- Sidebar settings ----------
st.sidebar.header("nQDI™ Controls")
alpha = st.sidebar.slider("HR exponent α (higher = HR matters more)", 0.5, 2.5, 1.5, 0.1)
smooth_span = st.sidebar.slider("EMA smoothing (seconds)", 1, 60, 20, 1)
np_window = st.sidebar.slider("NP window (seconds)", 10, 60, 30, 1)
dur_default = [15,30,60,300]
durations_secs = st.sidebar.multiselect("Best/worst durations (s)", options=[15,30,60,120,300,600,1200], default=dur_default)
save_lb = st.sidebar.checkbox("Save summary to local leaderboard", value=False)
anonymize_lb = st.sidebar.checkbox("Anonymize leaderboard entries", value=True)
unit = st.sidebar.selectbox("Weight unit", ["kg","lbs"], index=0)

# ---------- Main tabs ----------
tabs = st.tabs(["Compare Rides", "Demo Rides", "Leaderboard", "About / Launch"])
tab_cmp, tab_demo, tab_lb, tab_about = tabs

with tab_cmp:
    st.header("Compare two .fit rides (FIT-only, stops removed by speed==0)")
    left, right = st.columns(2)
    with left:
        you_file = st.file_uploader("Your .fit", type=["fit"], key="you_fit")
        you_weight = st.number_input("Your weight (optional)", min_value=0.0, value=0.0, step=0.1, key="you_weight")
        you_lthr = st.number_input("Your LTHR (optional)", min_value=0, value=0, step=1, key="you_lthr")
    with right:
        friend_file = st.file_uploader("Friend .fit", type=["fit"], key="friend_fit")
        friend_weight = st.number_input("Friend weight (optional)", min_value=0.0, value=0.0, step=0.1, key="friend_weight")
        friend_lthr = st.number_input("Friend LTHR (optional)", min_value=0, value=0, step=1, key="friend_lthr")
    run = st.button("⚡ Compute nQDI™ & Roast (make it mean)")

    if run:
        if you_file is None or friend_file is None:
            st.error("Upload BOTH .fit files first. No files = no roast.")
        else:
            try:
                df_you = parse_fit_bytes(you_file.read())
                df_friend = parse_fit_bytes(friend_file.read())
            except Exception as e:
                st.exception(f"Failed to parse FIT: {e}")
                st.stop()

            if df_you.empty or df_friend.empty:
                st.error("No moving data after filtering stopped time. Check files.")
                st.stop()

            # Align by sec with inner join; fallback to truncation if no overlap
            merged = pd.merge(df_you, df_friend, on="sec", suffixes=("_you","_friend"))
            if merged.empty:
                min_len = min(len(df_you), len(df_friend))
                merged = pd.concat([df_you.iloc[:min_len].reset_index(drop=True), df_friend.iloc[:min_len].reset_index(drop=True)], axis=1)
                # ensure column names
                if "power" in merged.columns and "power" in merged.columns:
                    # rename to expected names if needed
                    merged.columns = ["sec","power_you","hr_you","speed_you","cadence_you","sec2","power_friend","hr_friend","speed_friend","cadence_friend"]
                    merged = merged.drop(columns=["sec2"])

            hr_you_present = (merged["hr_you"] > 0).any()
            hr_friend_present = (merged["hr_friend"] > 0).any()
            if not hr_you_present or not hr_friend_present:
                st.warning("⚠️ HR missing for one or both riders — missing HR sections use power-only fallback. Results less HR-accurate but ridicule remains.")

            you_lthr_val = you_lthr if you_lthr > 0 else None
            friend_lthr_val = friend_lthr if friend_lthr > 0 else None

            # effective watts
            y_eff = effective_watts_hr(merged["power_you"].to_numpy(), merged["hr_you"].to_numpy(), you_lthr_val, alpha=alpha)
            f_eff = effective_watts_hr(merged["power_friend"].to_numpy(), merged["hr_friend"].to_numpy(), friend_lthr_val, alpha=alpha)

            # smoothing
            y_eff_s = pd.Series(y_eff).ewm(span=smooth_span, adjust=False).mean().to_numpy()
            f_eff_s = pd.Series(f_eff).ewm(span=smooth_span, adjust=False).mean().to_numpy()

            # NP-like time series
            y_np_ts = ts_normalized_power(y_eff_s, window=np_window)
            f_np_ts = ts_normalized_power(f_eff_s, window=np_window)

            eps = 1e-6
            nqdi_raw = np.clip(np.divide(f_np_ts, np.maximum(y_np_ts, eps)), 0.0, 10.0)

            # weight unit conversion
            if unit == "lbs":
                you_w_kg = you_weight * 0.45359237 if you_weight > 0 else 0.0
                friend_w_kg = friend_weight * 0.45359237 if friend_weight > 0 else 0.0
            else:
                you_w_kg = you_weight
                friend_w_kg = friend_weight

            if you_w_kg > 0 and friend_w_kg > 0:
                y_np_wkg = y_np_ts / you_w_kg
                f_np_wkg = f_np_ts / friend_w_kg
                nqdi_wkg = np.clip(np.divide(f_np_wkg, np.maximum(y_np_wkg, eps)), 0.0, 10.0)
            else:
                nqdi_wkg = nqdi_raw

            # summaries & scalar NP
            mean_raw = float(np.mean(nqdi_raw))
            mean_wkg = float(np.mean(nqdi_wkg))
            p95_raw = float(np.percentile(nqdi_raw, 95))

            you_np_scalar = scalar_normalized_power(y_eff_s, window=np_window)
            friend_np_scalar = scalar_normalized_power(f_eff_s, window=np_window)
            you_avg_power = float(np.mean(merged["power_you"]))
            friend_avg_power = float(np.mean(merged["power_friend"]))
            you_avg_hr = float(np.mean(merged["hr_you"])) if hr_you_present else None
            friend_avg_hr = float(np.mean(merged["hr_friend"])) if hr_friend_present else None

            # best efforts (effective watts)
            durations_be = [5,10,15,30,60,120,300,600,1200]
            be_you = best_effort_series(y_eff_s, durations_be)
            be_friend = best_effort_series(f_eff_s, durations_be)

            # best/worst QDI windows
            bw_qdi = compute_best_worst_windows(nqdi_wkg, durations_secs if durations_secs else [30,60,300])

            # extra metrics: Spite Index, FTP Stability, fake metrics
            spite_index = max(0.0, friend_np_scalar - you_np_scalar)
            ftp_stability = friend_np_scalar / max(you_np_scalar, 1e-3)
            sock_efficiency = round(random.uniform(0.2, 1.2), 2)
            komfort_ratio = round((friend_np_scalar / max(you_np_scalar, 1e-3)) * random.uniform(0.8,1.2), 2)
            quad_bank_timer = f"{random.randint(0,10):02d}:{random.randint(0,59):02d}"

            # display metrics
            a,b,c,d = st.columns(4)
            a.metric("💀 mean nQDI (raw)", f"{mean_raw:.3f}")
            b.metric("⚖ mean nQDI (W/kg)", f"{mean_wkg:.3f}")
            c.metric("🔪 peak nQDI (95%)", f"{p95_raw:.3f}")
            d.metric("🧮 α (HR weight)", f"{alpha:.2f}")

            st.markdown("### Rider summaries")
            left_col, right_col = st.columns(2)
            with left_col:
                st.write("**You**")
                st.write(f"- Avg power: {you_avg_power:.1f} W")
                st.write(f"- NP-like (eff): {you_np_scalar:.1f} W")
                if you_avg_hr is not None:
                    st.write(f"- Avg HR: {you_avg_hr:.0f} bpm")
                if you_w_kg > 0:
                    st.write(f"- Weight: {you_w_kg:.1f} kg → {you_np_scalar/you_w_kg:.2f} W/kg")
            with right_col:
                st.write("**Friend**")
                st.write(f"- Avg power: {friend_avg_power:.1f} W")
                st.write(f"- NP-like (eff): {friend_np_scalar:.1f} W")
                if friend_avg_hr is not None:
                    st.write(f"- Avg HR: {friend_avg_hr:.0f} bpm")
                if friend_w_kg > 0:
                    st.write(f"- Weight: {friend_w_kg:.1f} kg → {friend_np_scalar/friend_w_kg:.2f} W/kg")

            # best efforts table
            st.markdown("### 🔥 Best efforts (effective watts)")
            be_df = pd.DataFrame({"You": be_you, "Friend": be_friend})
            st.table(be_df.T)

            # best/worst QDI windows table
            st.markdown("### 📌 Best / Worst QDI windows")
            bw_rows = []
            for e in bw_qdi:
                bw_rows.append({
                    "duration_s": e["duration"],
                    "best_nqdi": e["best_val"],
                    "best_end_sec": e["best_idx"],
                    "worst_nqdi": e["worst_val"],
                    "worst_end_sec": e["worst_idx"]
                })
            st.table(pd.DataFrame(bw_rows))

            # roast
            roast_msg = choose_roast(mean_wkg)
            st.markdown("### 🔥 Roast (copy to group chat and watch drinks depart mouths)")
            st.info(roast_msg)

            # leaderboard save
            if save_lb:
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "you_file": getattr(you_file, "name", "you.fit") if not anonymize_lb else "you.fit",
                    "friend_file": getattr(friend_file, "name", "friend.fit") if not anonymize_lb else "friend.fit",
                    "mean_nqdi_raw": mean_raw,
                    "mean_nqdi_wkg": mean_wkg,
                    "you_np": you_np_scalar,
                    "friend_np": friend_np_scalar,
                    "spite_index": spite_index,
                    "ftp_stability": ftp_stability
                }
                save_entry(entry, anonymize=anonymize_lb)
                st.success("Saved to leaderboard (local CSV)")

            # report generation & download
            report_lines = []
            report_lines.append("nQDI™ Roast Report")
            report_lines.append(f"Generated (UTC): {datetime.utcnow().isoformat()}")
            report_lines.append(f"You file: {getattr(you_file,'name','you.fit') if not anonymize_lb else 'you.fit'}")
            report_lines.append(f"Friend file: {getattr(friend_file,'name','friend.fit') if not anonymize_lb else 'friend.fit'}")
            report_lines.append("")
            report_lines.append(f"Mean nQDI (raw): {mean_raw:.3f}")
            report_lines.append(f"Mean nQDI (W/kg): {mean_wkg:.3f}")
            report_lines.append(f"Peak nQDI (95%): {p95_raw:.3f}")
            report_lines.append("")
            report_lines.append("You — stats:")
            report_lines.append(f"  Avg power: {you_avg_power:.1f} W")
            report_lines.append(f"  NP-like (eff): {you_np_scalar:.1f} W")
            if you_avg_hr is not None: report_lines.append(f"  Avg HR: {you_avg_hr:.0f} bpm")
            if you_w_kg > 0: report_lines.append(f"  Weight(kg): {you_w_kg:.1f}")
            report_lines.append("")
            report_lines.append("Friend — stats:")
            report_lines.append(f"  Avg power: {friend_avg_power:.1f} W")
            report_lines.append(f"  NP-like (eff): {friend_np_scalar:.1f} W")
            if friend_avg_hr is not None: report_lines.append(f"  Avg HR: {friend_avg_hr:.0f} bpm")
            if friend_w_kg > 0: report_lines.append(f"  Weight(kg): {friend_w_kg:.1f}")
            report_lines.append("")
            report_lines.append("Best efforts (You):")
            for k,v in be_you.items():
                report_lines.append(f"  {k}s: {'' if v is None else f'{v:.1f} W'}")
            report_lines.append("Best efforts (Friend):")
            for k,v in be_friend.items():
                report_lines.append(f"  {k}s: {'' if v is None else f'{v:.1f} W'}")
            report_lines.append("")
            report_lines.append("Extra metrics:")
            report_lines.append(f"  Spite Index™: {spite_index:.1f} W")
            report_lines.append(f"  FTP Stability Index™: {ftp_stability:.2f}")
            report_lines.append(f"  Sock Height Efficiency Index™: {sock_efficiency}")
            report_lines.append(f"  KOMfort Ratio™: {komfort_ratio}")
            report_lines.append(f"  Quad Bankruptcy Timer™ (approx): {quad_bank_timer}")
            report_lines.append("")
            report_lines.append("Roast:")
            report_lines.append(roast_msg)
            report_text = "\n".join(report_lines)
            st.download_button("Download roast report (txt)", data=report_text, file_name="nqdi_roast_report.txt", mime="text/plain")

            # PLOTS: highlight best/worst QDI windows
            st.markdown("### 📈 Plots")
            t = np.arange(len(nqdi_raw))
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                specs=[[{"secondary_y": True}], [{}], [{}]],
                                subplot_titles=("Power & HR", "NP-like (effective watts)", "nQDI (raw & W/kg)"))

            # Top: power & HR
            fig.add_trace(go.Scatter(x=t, y=merged["power_you"], name="You Power (W)"), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=merged["power_friend"], name="Friend Power (W)"), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=merged["hr_you"], name="You HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(x=t, y=merged["hr_friend"], name="Friend HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Power (W)", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="HR (bpm)", row=1, col=1, secondary_y=True)

            # Mid: NP-like
            fig.add_trace(go.Scatter(x=t, y=y_np_ts, name="You NP-like (eff W)"), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=f_np_ts, name="Friend NP-like (eff W)"), row=2, col=1)
            fig.update_yaxes(title_text="NP-like (W)", row=2, col=1)

            # Bottom: nQDI
            fig.add_trace(go.Scatter(x=t, y=nqdi_raw, name="nQDI Raw"), row=3, col=1)
            fig.add_trace(go.Scatter(x=t, y=nqdi_wkg, name="nQDI W/kg"), row=3, col=1)
            fig.add_hline(y=1.0, line_dash="dash", annotation_text="Break-even", row=3, col=1)

            # Highlight best/worst windows from bw_qdi
            for e in bw_qdi:
                d = e["duration"]
                if e["best_idx"] is not None:
                    x0 = max(0, e["best_idx"] - d + 1)
                    x1 = e["best_idx"]
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="green", opacity=0.08, line_width=0, row=3, col=1)
                if e["worst_idx"] is not None:
                    x0 = max(0, e["worst_idx"] - d + 1)
                    x1 = e["worst_idx"]
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="red", opacity=0.08, line_width=0, row=3, col=1)

            # Distribution histogram (small separate figure to the right)
            fig.update_layout(height=900, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)

            # QDI distribution
            st.markdown("### 🧨 QDI Distribution")
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=nqdi_wkg, nbinsx=40))
            hist_fig.update_layout(height=300, xaxis_title="nQDI (W/kg)", yaxis_title="Count")
            st.plotly_chart(hist_fig, use_container_width=True)

with tab_demo:
    st.header("Demo rides (generate screenshots, no real friends harmed)")
    if st.button("Generate a set of demo rides"):
        rng = np.random.default_rng(42)
        t = np.arange(1800)  # 30 minutes
        base = 220 + 30 * np.sin(t/80) + rng.normal(0,10,size=t.size)
        friend = base * (1 + 0.12 * np.sin(t/200)) + rng.normal(0,8,size=t.size)
        you_df = pd.DataFrame({"sec":t,"power":base,"hr":150 + 6*np.sin(t/40)+rng.normal(0,3,size=t.size),"speed":8+rng.normal(0,0.3,size=t.size),"cadence":85+rng.normal(0,5,size=t.size)})
        friend_df = pd.DataFrame({"sec":t,"power":friend,"hr":145 + 4*np.sin(t/50)+rng.normal(0,2,size=t.size),"speed":8.2+rng.normal(0,0.2,size=t.size),"cadence":92+rng.normal(0,3,size=t.size)})
        st.success("Demo rides generated — use these for screenshots")
        st.line_chart(you_df[["power","hr"]].rename(columns={"power":"Power (W)","hr":"HR (bpm)"}).iloc[:600])
        st.line_chart(friend_df[["power","hr"]].rename(columns={"power":"Power (W)","hr":"HR (bpm)"}).iloc[:600])

with tab_lb:
    st.header("Leaderboard")
    lb = load_leaderboard()
    if lb.empty:
        st.info("Leaderboard is empty. Enable saving to populate it.")
    else:
        st.dataframe(lb.sort_values("timestamp", ascending=False).reset_index(drop=True))
        if st.button("Clear leaderboard"):
            try:
                os.remove(LEADERBOARD_CSV)
                st.success("Leaderboard cleared.")
            except Exception as e:
                st.error(f"Could not clear: {e}")

with tab_about:
    st.header("About, privacy & launch checklist")
    st.markdown("""
**What this does:** FIT-only. Removes stopped time (speed==0). Computes HR-dominant effective watts (lower HR = more efficient), NP-style math, per-second nQDI and W/kg, lots of roasts, and exportable reports.

**Privacy:** Files are processed in-memory. If you enable leaderboard saving, a small summary CSV will be written to the app directory. When deploying publicly, use a db or opt-in explicit consent.

**Launch checklist:**
- Add requirements.txt with: streamlit pandas numpy plotly fitparse
- Consider Strava OAuth for automatic pulls.
- Replace local CSV with DB for durable leaderboard on cloud.
""")
    st.markdown("### Suggested requirements.txt")
    st.code("\n".join(["streamlit>=1.24.0","pandas>=2.0.0","numpy>=1.24.0","plotly>=5.0.0","fitparse>=1.2.0"]))

st.caption("nQDI™: scientific roast therapy. Now go share responsibly (or not).")

