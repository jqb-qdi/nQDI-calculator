# app.py
"""
nQDI™ Live — Polished launch-ready Streamlit app (FIT-only)
Features:
 - FIT-only parsing (power/hr/speed/cadence), removes stopped time (speed==0)
 - HR-aware effective watts (lower HR @ same power => better)
 - Per-second NP-style series and scalar NP
 - Best-efforts table (many durations)
 - Best/worst QDI windows for selected durations
 - Weight optional; support for kg or lbs (single unit for both)
 - Demo synthetic fit generator for screenshots
 - Rotating escalating roasts (lots of them)
 - Optional anonymized leaderboard CSV save
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

# ---------------------------
# Config & UI niceties
# ---------------------------
LEADERBOARD_CSV = "nqdi_leaderboard.csv"
st.set_page_config(page_title="nQDI™ Live — Ready to Roast", layout="wide")
st.title("🩸 nQDI™ Live — Quad Deficit Index (QDI) — Final Polished Build")
st.caption("Because power meters, Strava KOM tears, and ✨aërø✨ socks deserve a tribunal. Lower HR at same power = you’re stronger. QDI tells the truth.")

# ---------------------------
# Helpers: FIT parsing & cleaning
# ---------------------------
@st.cache_data
def parse_fit_bytes(bytes_io):
    """Parse FIT bytes into a DataFrame with sec, power, hr, speed, cadence.
       Remove stopped time (speed == 0)."""
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
    # drop rows without power entirely (no point)
    df = df.dropna(subset=["power"]).copy()
    # if timestamp exists, resample to 1s and build sec
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
        # fabricate sec
        df = df.reset_index(drop=True)
        df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
        df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
        df["speed"] = pd.to_numeric(df.get("speed", 0), errors="coerce").fillna(0)
        df["cadence"] = pd.to_numeric(df.get("cadence", 0), errors="coerce").fillna(0)
        df["sec"] = np.arange(len(df))
    # convert speed nan->0 and filter stopped slices (speed == 0)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    df = df[df["speed"] > 0]  # only moving time
    # finalize numeric columns
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce").fillna(0)
    df = df.reset_index(drop=True)
    return df[["sec","power","hr","speed","cadence"]]

# ---------------------------
# Math helpers (NP, effective watts, best efforts)
# ---------------------------
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
    """Per-second NP-like series (uses rolling mean then rolling p4 mean ^0.25)."""
    s = pd.Series(power_arr).astype(float)
    if len(s) == 0:
        return np.array([])
    rm = s.rolling(window=window, min_periods=1).mean()
    p4 = (rm ** 4).rolling(window=window, min_periods=1).mean()
    np_ts = np.power(p4, 0.25).to_numpy()
    return np.nan_to_num(np_ts, nan=np.nanmean(np_ts) if np.any(~np.isnan(np_ts)) else s.to_numpy())

def effective_watts_hr(power_arr, hr_arr, lthr, alpha=1.5, clip_low=0.3, clip_high=3.0):
    power = np.array(power_arr, dtype=float)
    hr = np.array(hr_arr, dtype=float)
    # if no useful HR or no LTHR -> return raw power (don’t invent HR)
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

def compute_best_worst_qdi(qdi_series, durations):
    return compute_best_worst_windows(qdi_series, durations)

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

# ---------------------------
# Roasts (expanded, escalating)
# ---------------------------
ROAST_BUCKETS = {
    "elite": [
        "🚀 Overlord: Your friend’s legs filed a motion to adopt your KOMs as decorations.",
        "✨ Peerless efficiency: your friend could nap and still outsprint you."
    ],
    "close": [
        "⚡ Tight duel — both of you look frighteningly committed to carbs.",
        "😅 Neck and neck: equal parts suffering and dignity."
    ],
    "meh": [
        "💥 Mildly embarrassed: your watts shivered and considered retirement.",
        "🤡 Someone brought intervals to a cake ride. Not you, sadly."
    ],
    "bad": [
        "🔥 Leaky tire apocalypse: your pedaling resembles a sad trombone.",
        "📉 Your FTP chart now qualifies as modern art (abstract despair)."
    ],
    "nuclear": [
        "💣 Quad Bankruptcy: please accept this coupon for group therapy.",
        "⚰️ Instant extinction: your Strava now a cautionary tale."
    ]
}
MICRO_ROASTS = [
    "That sprint at t=42s? Spaghetti limbs detonated.",
    "HR curve reads like a horror short film; watts hid under a rock.",
    "Peak humiliation archived — PNG ready for meme distribution.",
    "Cadence looked like someone trying to play drums badly.",
    "Strava auto-pause judged your life choices."
]

def choose_roast(mean_qdi_wkg):
    if mean_qdi_wkg is None or math.isnan(mean_qdi_wkg):
        core = "nQDI undefined — likely because someone uploaded a coffee break instead of a ride."
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
    return f"{core}  \n{micro}"

# ---------------------------
# Leaderboard helpers
# ---------------------------
def save_entry(entry, csv_path=LEADERBOARD_CSV, anonymize=True):
    df_new = pd.DataFrame([entry])
    if anonymize:
        # mask file names to just basenames (not paths). also optionally remove names.
        df_new = df_new.rename(columns=lambda c: c)
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

# ---------------------------
# Sidebar / config
# ---------------------------
st.sidebar.header("nQDI™ Launch Controls")
alpha = st.sidebar.slider("HR exponent α (higher = HR matters more)", 0.5, 2.5, 1.5, 0.1)
smooth_span = st.sidebar.slider("EMA smoothing (span seconds)", 1, 60, 20, 1)
np_window = st.sidebar.slider("NP window (seconds)", 10, 60, 30, 1)
dur_default = [15,30,60,300]
durations_secs = st.sidebar.multiselect("Best/worst durations (s)", options=[15,30,60,120,300,600,1200], default=dur_default)
save_lb = st.sidebar.checkbox("Enable saving to local leaderboard CSV", value=False)
anonymize_lb = st.sidebar.checkbox("Anonymize leaderboard entries (hide filenames)", value=True)
unit = st.sidebar.selectbox("Weight unit for inputs", ["kg","lbs"], index=0)

# ---------------------------
# Main: tabs & UI
# ---------------------------
tabs = st.tabs(["Compare Rides", "Demo Ride", "Leaderboard", "About / Launch Checklist"])
tab_cmp, tab_demo, tab_lb, tab_about = tabs

with tab_cmp:
    st.header("Compare two `.fit` rides — FIT only (speed==0 removed)")
    col1, col2 = st.columns(2)
    with col1:
        you_file = st.file_uploader("Your .fit", type=["fit"], key="you_fit")
        you_weight = st.number_input("Your weight (optional)", min_value=0.0, value=0.0, step=0.1, key="you_weight")
        you_lthr = st.number_input("Your LTHR (optional)", min_value=0, value=0, step=1, key="you_lthr")
    with col2:
        friend_file = st.file_uploader("Friend .fit", type=["fit"], key="friend_fit")
        friend_weight = st.number_input("Friend weight (optional)", min_value=0.0, value=0.0, step=0.1, key="friend_weight")
        friend_lthr = st.number_input("Friend LTHR (optional)", min_value=0, value=0, step=1, key="friend_lthr")

    run_btn = st.button("⚡ Compute nQDI™ & Roast")

    if run_btn:
        if (you_file is None) or (friend_file is None):
            st.error("Upload BOTH .fit files. No files → no roast (and no laughs).")
        else:
            try:
                df_you = parse_fit_bytes(you_file.read())
                df_friend = parse_fit_bytes(friend_file.read())
            except Exception as e:
                st.exception(f"FIT parse failed: {e}")
                st.stop()

            if df_you.empty or df_friend.empty:
                st.error("After removing stopped time there is no moving data in one of the files. Check your .fit or try a different ride.")
                st.stop()

            # Align by sec via inner join to match real timestamps
            merged = pd.merge(df_you, df_friend, on="sec", suffixes=("_you","_friend"))
            if merged.empty:
                # fallback: align by truncation if timestamps don't overlap
                min_len = min(len(df_you), len(df_friend))
                merged = pd.concat([df_you.iloc[:min_len].reset_index(drop=True), df_friend.iloc[:min_len].reset_index(drop=True)], axis=1)
                # rename expected columns if necessary
                if "power" in merged.columns and "power" in merged.columns:
                    merged.columns = [c if not c.endswith("_friend") else c for c in merged.columns]

            # Prepare arrays
            hr_you_present = (merged["hr_you"] > 0).any()
            hr_friend_present = (merged["hr_friend"] > 0).any()
            if not hr_you_present or not hr_friend_present:
                st.warning("⚠️ Heart rate missing for one or both riders — those sections use power-only fallback. The nQDI™ will be less HR-accurate but still savage.")

            you_lthr_val = you_lthr if you_lthr > 0 else None
            friend_lthr_val = friend_lthr if friend_lthr > 0 else None

            # Effective watts (HR-dominant when possible)
            y_eff = effective_watts_hr(merged["power_you"].to_numpy(), merged["hr_you"].to_numpy(), you_lthr_val, alpha=alpha)
            f_eff = effective_watts_hr(merged["power_friend"].to_numpy(), merged["hr_friend"].to_numpy(), friend_lthr_val, alpha=alpha)

            # Smoothing
            y_eff_s = pd.Series(y_eff).ewm(span=smooth_span, adjust=False).mean().to_numpy()
            f_eff_s = pd.Series(f_eff).ewm(span=smooth_span, adjust=False).mean().to_numpy()

            # NP-like per-second series
            y_np_ts = ts_normalized_power(y_eff_s, window=np_window)
            f_np_ts = ts_normalized_power(f_eff_s, window=np_window)

            eps = 1e-6
            nqdi_raw = np.clip(np.divide(f_np_ts, np.maximum(y_np_ts, eps)), 0.0, 10.0)

            # Weight conversion if needed
            if unit == "lbs":
                you_w_kg = you_weight * 0.45359237 if you_weight > 0 else 0.0
                friend_w_kg = friend_weight * 0.45359237 if friend_weight > 0 else 0.0
            else:
                you_w_kg = you_weight
                friend_w_kg = friend_weight

            # weight-adjusted QDI
            if you_w_kg > 0 and friend_w_kg > 0:
                y_np_wkg = y_np_ts / you_w_kg
                f_np_wkg = f_np_ts / friend_w_kg
                nqdi_wkg = np.clip(np.divide(f_np_wkg, np.maximum(y_np_wkg, eps)), 0.0, 10.0)
            else:
                nqdi_wkg = nqdi_raw  # fallback if weights missing

            # Summaries
            mean_raw = float(np.mean(nqdi_raw))
            mean_wkg = float(np.mean(nqdi_wkg))
            p95_raw = float(np.percentile(nqdi_raw, 95))

            you_np_scalar = scalar_normalized_power(y_eff_s, window=np_window)
            friend_np_scalar = scalar_normalized_power(f_eff_s, window=np_window)
            you_avg_power = float(np.mean(merged["power_you"]))
            friend_avg_power = float(np.mean(merged["power_friend"]))
            you_avg_hr = float(np.mean(merged["hr_you"])) if hr_you_present else None
            friend_avg_hr = float(np.mean(merged["hr_friend"])) if hr_friend_present else None

            # Best efforts (effective watts)
            durations_be = [5,10,15,30,60,120,300,600,1200]
            be_you = best_effort_series(y_eff_s, durations_be)
            be_friend = best_effort_series(f_eff_s, durations_be)

            # Best/worst QDI windows for selected durations
            bw_qdi = compute_best_worst_windows(nqdi_wkg, durations_secs if durations_secs else [30,60,300])

            # Output metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("💀 mean nQDI (raw)", f"{mean_raw:.3f}")
            c2.metric("⚖ mean nQDI (W/kg)", f"{mean_wkg:.3f}")
            c3.metric("🔪 peak nQDI (95%)", f"{p95_raw:.3f}")
            c4.metric("🧮 α (HR weight)", f"{alpha:.2f}")

            st.markdown("### Rider summaries")
            left, right = st.columns(2)
            with left:
                st.write("**You**")
                st.write(f"- Avg power: {you_avg_power:.1f} W")
                st.write(f"- NP-like (eff): {you_np_scalar:.1f} W")
                if you_avg_hr is not None:
                    st.write(f"- Avg HR: {you_avg_hr:.0f} bpm")
                if you_w_kg > 0:
                    st.write(f"- Weight: {you_w_kg:.1f} kg → {you_np_scalar/you_w_kg:.2f} W/kg")
            with right:
                st.write("**Friend**")
                st.write(f"- Avg power: {friend_avg_power:.1f} W")
                st.write(f"- NP-like (eff): {friend_np_scalar:.1f} W")
                if friend_avg_hr is not None:
                    st.write(f"- Avg HR: {friend_avg_hr:.0f} bpm")
                if friend_w_kg > 0:
                    st.write(f"- Weight: {friend_w_kg:.1f} kg → {friend_np_scalar/friend_w_kg:.2f} W/kg")

            # Best efforts table
            be_df = pd.DataFrame({"You": be_you, "Friend": be_friend})
            st.markdown("### 🔥 Best efforts (effective watts)")
            st.table(be_df.T)

            # Best/worst QDI windows
            bw_rows = []
            for entry in bw_qdi:
                d = entry["duration"]
                bw_rows.append({
                    "duration_s": d,
                    "best_nqdi": entry["best_val"],
                    "best_end_sec": entry["best_idx"],
                    "worst_nqdi": entry["worst_val"],
                    "worst_end_sec": entry["worst_idx"],
                })
            st.markdown("### 📌 Best / Worst QDI windows")
            st.table(pd.DataFrame(bw_rows))

            # Roast
            roast_msg = choose_roast(mean_wkg)
            st.markdown("### 🔥 Roast (share this and make people spit their drinks)")
            st.info(roast_msg)

            # Optionally save to leaderboard
            if save_lb:
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "you_file": getattr(you_file, "name", "you.fit") if not anonymize_lb else "you.fit",
                    "friend_file": getattr(friend_file, "name", "friend.fit") if not anonymize_lb else "friend.fit",
                    "mean_nqdi_raw": mean_raw,
                    "mean_nqdi_wkg": mean_wkg,
                    "you_np": you_np_scalar,
                    "friend_np": friend_np_scalar
                }
                save_entry(entry, anonymize=anonymize_lb)
                st.success("Saved summary to leaderboard.")

            # Roast report download
            report = []
            report.append("nQDI™ Roast Report")
            report.append(f"Generated (UTC): {datetime.utcnow().isoformat()}")
            report.append(f"You file: {getattr(you_file,'name','you.fit') if not anonymize_lb else 'you.fit'}")
            report.append(f"Friend file: {getattr(friend_file,'name','friend.fit') if not anonymize_lb else 'friend.fit'}")
            report.append("")
            report.append(f"Mean nQDI (raw): {mean_raw:.3f}")
            report.append(f"Mean nQDI (W/kg): {mean_wkg:.3f}")
            report.append(f"Peak nQDI (95%): {p95_raw:.3f}")
            report.append("")
            report.append("You — stats:")
            report.append(f"  Avg power: {you_avg_power:.1f} W")
            report.append(f"  NP-like (eff): {you_np_scalar:.1f} W")
            if you_avg_hr is not None: report.append(f"  Avg HR: {you_avg_hr:.0f} bpm")
            if you_w_kg > 0: report.append(f"  Weight(kg): {you_w_kg:.1f}")
            report.append("")
            report.append("Friend — stats:")
            report.append(f"  Avg power: {friend_avg_power:.1f} W")
            report.append(f"  NP-like (eff): {friend_np_scalar:.1f} W")
            if friend_avg_hr is not None: report.append(f"  Avg HR: {friend_avg_hr:.0f} bpm")
            if friend_w_kg > 0: report.append(f"  Weight(kg): {friend_w_kg:.1f}")
            report.append("")
            report.append("Best efforts (You):")
            for k,v in be_you.items():
                report.append(f"  {k}s: {'' if v is None else f'{v:.1f} W'}")
            report.append("Best efforts (Friend):")
            for k,v in be_friend.items():
                report.append(f"  {k}s: {'' if v is None else f'{v:.1f} W'}")
            report.append("")
            report.append("Roast:")
            report.append(roast_msg)
            report_txt = "\n".join(report)
            st.download_button("Download roast report (txt)", data=report_txt, file_name="nqdi_roast_report.txt", mime="text/plain")

            # Plots
            st.markdown("### 📈 Plots (hover & zoom to relive the suffering)")
            t = np.arange(len(nqdi_raw))
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                specs=[[{"secondary_y": True}], [{}], [{}]],
                                subplot_titles=("Power & HR", "NP-like (effective watts)", "nQDI (raw & W/kg)"))
            # Top
            fig.add_trace(go.Scatter(x=t, y=merged["power_you"], name="You Power (W)"), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=merged["power_friend"], name="Friend Power (W)"), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=merged["hr_you"], name="You HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(x=t, y=merged["hr_friend"], name="Friend HR (bpm)", line=dict(dash="dot")), row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Power (W)", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="HR (bpm)", row=1, col=1, secondary_y=True)
            # mid
            fig.add_trace(go.Scatter(x=t, y=y_np_ts, name="You NP-like (eff W)"), row=2, col=1)
            fig.add_trace(go.Scatter(x=t, y=f_np_ts, name="Friend NP-like (eff W)"), row=2, col=1)
            fig.update_yaxes(title_text="NP-like (W)", row=2, col=1)
            # bottom
            fig.add_trace(go.Scatter(x=t, y=nqdi_raw, name="nQDI Raw"), row=3, col=1)
            fig.add_trace(go.Scatter(x=t, y=nqdi_wkg, name="nQDI W/kg"), row=3, col=1)
            fig.add_hline(y=1.0, line_dash="dash", annotation_text="Break-even", row=3, col=1)
            fig.update_yaxes(title_text="nQDI", row=3, col=1)
            fig.update_layout(height=900, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)

with tab_demo:
    st.header("Demo ride (quick screenshot/demo)")
    if st.button("Generate demo rides"):
        # generate two synthetic rides for screenshot/demo
        np.random.seed(42)
        t = np.arange(1800)  # 30 min
        base_you = 200 + 30 * np.sin(t/60) + np.random.normal(0, 10, size=t.size)
        base_friend = base_you * (1 + 0.12 * np.sin(t/120)) + np.random.normal(0,8,size=t.size)
        you_df = pd.DataFrame({"sec": t, "power": base_you, "hr": 150 + 5*np.sin(t/40)+np.random.normal(0,3,size=t.size), "speed": 8 + np.random.normal(0,0.5,size=t.size), "cadence": 85 + np.random.normal(0,5,size=t.size)})
        friend_df = pd.DataFrame({"sec": t, "power": base_friend, "hr": 145 + 3*np.sin(t/50)+np.random.normal(0,2,size=t.size), "speed": 8.2 + np.random.normal(0,0.4,size=t.size), "cadence": 92 + np.random.normal(0,4,size=t.size)})
        st.success("Demo rides generated — use them for screenshotting the roast report if you're a monster.")
        st.write("You (preview):")
        st.line_chart(you_df[["power","hr"]].rename(columns={"power":"Power (W)","hr":"HR (bpm)"}).iloc[:600])
        st.write("Friend (preview):")
        st.line_chart(friend_df[["power","hr"]].rename(columns={"power":"Power (W)","hr":"HR (bpm)"}).iloc[:600])

with tab_lb:
    st.header("Local Leaderboard (summaries)")
    lb = load_leaderboard()
    if lb.empty:
        st.info("Leaderboard empty. Enable saving and run a comparison to populate.")
    else:
        st.dataframe(lb.sort_values("timestamp", ascending=False).reset_index(drop=True))
        if st.button("Clear leaderboard"):
            try:
                os.remove(LEADERBOARD_CSV)
                st.success("Leaderboard cleared.")
            except Exception as e:
                st.error(f"Could not clear: {e}")

with tab_about:
    st.header("About & Launch Checklist")
    st.markdown("""
**Polished launch checklist (quick):**
- [ ] `requirements.txt` in repo (see below).
- [ ] Decide whether to enable leaderboard persistence (privacy!).
- [ ] If deploying to Streamlit Cloud, remember local CSV will not persist across container restarts — use an external DB or object storage.
- [ ] Add a short, obvious consent line near uploader: "By uploading, you consent to data processing and possible sharing with this group's chat for roasting."
- [ ] Optional: Hook Strava OAuth so users can connect instead of uploading files.

**Quick tips to make people spit drinks:**
- Use the demo button to capture a savage screenshot.
- Post the roast report PNG/TXT to the group with the headline: "nQDI™ results — the roast is served."
- Encourage people to enable leaderboard saving and watch friendships melt.

**Essential dependencies**: `streamlit pandas numpy plotly fitparse`
""")
    st.markdown("### Recommended `requirements.txt`")
    st.code("\n".join([
        "streamlit>=1.24.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.0.0",
        "fitparse>=1.2.0"
    ]))

st.caption("nQDI™: Brilliantly petty, scientifically hilarious. Now go break some ego (preferably not permanently) — and don’t forget to wear your ✨aërø✨ socks for aerodynamic virtue signaling.")
