# nQDI_unhinged_fit_only.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile
import random

st.set_page_config(
    page_title="nQDI™ Live: Your Ego’s Funeral Service",
    layout="wide"
)

# ---------------------------
# FIT Reading
# ---------------------------
def read_fit(file_like):
    fitfile = FitFile(file_like)
    rows = []
    for msg in fitfile.get_messages("record"):
        rec = {"power": None, "hr": None, "speed": None, "cadence": None}
        for d in msg:
            if d.name == "power": rec["power"] = d.value
            elif d.name == "heart_rate": rec["hr"] = d.value
            elif d.name == "speed": rec["speed"] = d.value  # meters/second
            elif d.name == "cadence": rec["cadence"] = d.value
        rows.append(rec)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["power"]).reset_index(drop=True)

    # speed filter → remove stopped time (speed < 0.5 m/s ~ 1.1mph)
    df = df[df["speed"].fillna(0) > 0.5]

    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce").fillna(0)
    df["time"] = pd.to_timedelta(np.arange(len(df)), unit="s")
    return df[["time", "power", "hr", "speed", "cadence"]]

def load_file(upload):
    name = upload.name.lower()
    if not name.endswith(".fit"):
        raise ValueError("Unsupported file. Only .fit files are allowed. CSVs belong in spreadsheets, not in humiliation simulators.")
    return read_fit(io.BytesIO(upload.read()))

# ---------------------------
# Math Utilities
# ---------------------------
def ema(series, span=30):
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def effective_watts(power, hr, lthr, alpha=1.5):
    if lthr <= 0 or np.max(hr) == 0:  # no HR → ignore scaling
        return power
    hr_ratio = np.divide(lthr, np.maximum(hr, 1e-3))  # lower HR = better
    hr_ratio = np.clip(hr_ratio, 0.3, 3.0)
    return power * (hr_ratio ** alpha)

def align_to_same_length(a_df, b_df):
    merged = pd.merge(a_df, b_df, left_index=True, right_index=True, suffixes=("_you","_friend"))
    return merged

# ---------------------------
# Roasts
# ---------------------------
def roast_text(nqdi_kg, df_you, df_friend):
    lines = []
    if nqdi_kg < 0.85:
        lines.append("🚀 W/kg Overlord: Your quads are in god-tier mode. Expect your friend to fake a mechanical next ride.")
    elif nqdi_kg < 1.0:
        lines.append("⚡ Barely acceptable: You’re ahead, but HR says you’re one oat milk latte away from implosion.")
    elif nqdi_kg < 1.15:
        lines.append("💥 Mediocre collapse: Your watts are crying, your ego is doing CPR, and your cadence looks like blender-on-low.")
    elif nqdi_kg < 1.35:
        lines.append("🔥 Flaming disgrace: Pedaling like your crankset is made of Play-Doh. Your friend is basically on an e-bike now.")
    elif nqdi_kg < 1.8:
        lines.append("💀 Domestique doom: They KOM, you bottle-fetch. It’s tradition now.")
    else:
        lines.append("💣 Quad Deficit Armageddon: Retire. Take up crochet. Donate your bike to Goodwill. Light a candle for your FTP.")

    # Rotating insults based on cadence, HR, or speed
    micro = []
    if df_you["cadence"].mean() < 70:
        micro.append("Cadence report: You’re basically mashing corn tortillas at 62 rpm.")
    if df_you["cadence"].mean() > 105:
        micro.append("Cadence report: 112 rpm hamster mode, watts still trash.")
    if df_you["hr"].max() > 190:
        micro.append("Your HR hit 195 bpm. That wasn’t cycling, that was cardiac cosplay.")
    if df_friend["power"].max() > df_you["power"].max():
        micro.append("Friend’s sprint: atomic detonation. Your sprint: paper airplane.")
    if df_you["speed"].mean() < df_friend["speed"].mean():
        micro.append("Average speed gap detected. Were you drafting behind a parked car?")
    
    # fallback micro-roasts if none triggered
    fallback_micro = [
        "That sprint at t=42s? spaghetti limbs in freefall.",
        "Strava KOM dreams: vaporized.",
        "Even your Garmin auto-pause judged you.",
        "Your watts whispered 'please stop' halfway through.",
        "FTP stability index: clinically unstable."
    ]
    if not micro:
        micro = fallback_micro
    
    lines.append(random.choice(micro))
    return "\n".join(lines)

# ---------------------------
# UI
# ---------------------------
st.markdown("# 🩸 nQDI™ Live: The Public Execution of Your Ego")
st.markdown("**Disclaimer:** Uploading FIT files here is like signing a waiver for self-inflicted roast sessions. The algorithm will strip away your dignity, expose your watts-per-kilo shame, and casually remind you that your 'big sprint' was the watt-equivalent of a dying flashlight. Proceed if you dare.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("You")
    your_file = st.file_uploader("Upload YOUR .fit", type=["fit"], key="u_file")
    your_weight = st.number_input("Weight (kg, optional)", min_value=0.0, max_value=300.0, format="%.1f", key="u_weight")
    your_lthr = st.number_input("LTHR (bpm, optional)", min_value=0, max_value=220, step=1, key="u_lthr")

with col2:
    st.subheader("Friend")
    friend_file = st.file_uploader("Upload FRIEND .fit", type=["fit"], key="f_file")
    friend_weight = st.number_input("Weight (kg, optional)", min_value=0.0, max_value=300.0, format="%.1f", key="f_weight")
    friend_lthr = st.number_input("LTHR (bpm, optional)", min_value=0, max_value=220, step=1, key="f_lthr")

alpha = st.slider("HR weighting exponent α", 0.5, 2.5, 1.5, 0.1)
smooth = st.slider("Smoothing seconds (EMA)", 1, 60, 30, 1)

compute_btn = st.button("⚡ Roast Me With Science")

# ---------------------------
# Compute
# ---------------------------
if compute_btn:
    if not your_file or not friend_file:
        st.error("Both files required. No data, no roast, no humiliation.")
        st.stop()

    try:
        df_you = load_file(your_file)
        df_friend = load_file(friend_file)
    except Exception as e:
        st.error(f"File read error: {e}")
        st.stop()

    merged = align_to_same_length(df_you, df_friend)

    y_eff = effective_watts(merged['power_you'], merged['hr_you'], your_lthr, alpha)
    f_eff = effective_watts(merged['power_friend'], merged['hr_friend'], friend_lthr, alpha)

    y_eff_s = ema(y_eff, span=smooth)
    f_eff_s = ema(f_eff, span=smooth)

    nqdi_raw = f_eff_s / np.maximum(y_eff_s, 1e-6)
    if your_weight > 0 and friend_weight > 0:
        nqdi_kg = (f_eff_s/friend_weight) / np.maximum(y_eff_s/your_weight,1e-6)
    else:
        nqdi_kg = nqdi_raw

    nqdi_raw = np.clip(nqdi_raw, 0, 10)
    nqdi_kg = np.clip(nqdi_kg, 0, 10)

    st.metric("💀 nQDI™ Raw (mean)", f"{np.mean(nqdi_raw):.3f}")
    st.metric("⚖️ nQDI™ Weight-adjusted (mean)", f"{np.mean(nqdi_kg):.3f}")

    st.markdown("### 🔥 Brutal Roasts")
    st.info(roast_text(np.mean(nqdi_kg), df_you, df_friend))

    t = merged.index.to_numpy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Power & HR — pain theater", "nQDI™ Live — humiliation in real time"))
    fig.add_trace(go.Scatter(x=t, y=merged['power_you'], name="You Power", mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=merged['power_friend'], name="Friend Power", mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=merged['hr_you'], name="You HR", mode='lines', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=merged['hr_friend'], name="Friend HR", mode='lines', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=nqdi_raw, name="nQDI Raw", mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=nqdi_kg, name="nQDI W/kg", mode='lines'), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Break-even humiliation line", row=2, col=1)

    fig.update_yaxes(title_text="Power / HR", row=1, col=1)
    fig.update_yaxes(title_text="nQDI", row=2, col=1)
    fig.update_xaxes(title_text="Seconds")
    fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

    st.plotly_chart(fig, use_container_width=True)
