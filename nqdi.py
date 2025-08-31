# nQDI_unhinged_v2.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile
import random

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="nQDI™ Live: Watt-Crushing, Quad-Bankrupting, Ego-Annihilating Simulator™",
    layout="wide"
)

# ---------------------------
# Utilities
# ---------------------------
def read_fit(file_like):
    fitfile = FitFile(file_like)
    rows = []
    for msg in fitfile.get_messages("record"):
        rec = {"power": None, "hr": None}
        for d in msg:
            if d.name == "power": rec["power"] = d.value
            elif d.name == "heart_rate": rec["hr"] = d.value
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["power"])
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(method="ffill").fillna(0)
    df["time"] = pd.to_timedelta(np.arange(len(df)), unit="s")
    return df[["time","power","hr"]]

def read_csv(file_like):
    df = pd.read_csv(file_like)
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(method="ffill").fillna(0)
    df["time"] = pd.to_timedelta(np.arange(len(df)), unit="s")
    return df[["time","power","hr"]]

def load_file(upload):
    name = upload.name.lower()
    if name.endswith(".fit"): return read_fit(io.BytesIO(upload.read()))
    elif name.endswith(".csv"): return read_csv(upload)
    else: raise ValueError("Unsupported file. Upload .fit or .csv")

def ema(series, span=30):
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def effective_watts(power, hr, lthr, alpha=1.5):
    power = np.array(power)
    hr = np.array(hr)
    if lthr <= 0 or np.all(hr==0):
        # fallback: use power only if HR missing
        return power
    hr_safe = np.where(hr==0, np.nan, hr)
    hr_ratio = lthr / hr_safe
    hr_ratio = np.clip(hr_ratio, 0.3, 3.0)
    eff = power * (np.nan_to_num(hr_ratio, nan=1.0) ** alpha)
    return eff

def align_to_same_length(a_df, b_df):
    merged = pd.merge(a_df, b_df, left_index=True, right_index=True, suffixes=("_you","_friend"))
    return merged

def roast_text(nqdi_kg):
    lines = []
    # escalating unhinged roasts
    if nqdi_kg < 0.85:
        lines.append("🚀 Absolute overlord: your friend's quads just filed restraining orders against your ego.")
    elif nqdi_kg < 1.0:
        lines.append("⚡ Slightly tolerable: your HR is gasping, but your ego barely survives.")
    elif nqdi_kg < 1.15:
        lines.append("💥 Meh domination: watts tremble, quads weep silently.")
    elif nqdi_kg < 1.35:
        lines.append("🔥 Leaky tire apocalypse: pedal strokes betray human decency.")
    elif nqdi_kg < 1.8:
        lines.append("💀 Domestique purgatory: you haul bottles while they harvest KOMs.")
    else:
        lines.append("💣 Catastrophic quad bankruptcy: retire, take up knitting, cry into ✨håndmådë cøttøn tūbùlårs✨ at 11psi.")

    micro_roasts = [
        "That sprint at t=42s? spaghetti limbs exploded.",
        "HR curve like a horror movie, watts weeping quietly.",
        "Peak nQDI recorded; funeral for your ego scheduled.",
        "Strava KOM dreams evaporated; ghosts of FTP past haunt your files.",
        "Every pedal stroke whispers: 'why even try?'",
        "Your QDI screams louder than your legs ever could.",
        "Quads evicted from the ego apartment; rent overdue in watts."
    ]
    lines.append(random.choice(micro_roasts))
    return "\n".join(lines)

# ---------------------------
# UI
# ---------------------------
st.markdown("# 🩸 nQDI™ Live: Watt-Crushing, Quad-Bankrupting, Ego-Annihilating Simulator™")
st.markdown("**Disclaimer:** This tool will obliterate your ego, calculate every shred of wattage like a merciless robot, and roast your legs with escalating cruelty. Proceed if you enjoy exquisite suffering.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("You")
    your_file = st.file_uploader("Upload YOUR file", type=["fit","csv"], key="u_file")
    your_weight = st.number_input("Weight (lbs or kg)", min_value=0.0, max_value=300.0, format="%.1f", key="u_weight")
    your_lthr = st.number_input("LTHR (bpm, optional if HR missing)", min_value=0, max_value=220, step=1, key="u_lthr")

with col2:
    st.subheader("Friend")
    friend_file = st.file_uploader("Upload FRIEND file", type=["fit","csv"], key="f_file")
    friend_weight = st.number_input("Weight (lbs or kg, same unit as yours)", min_value=0.0, max_value=300.0, format="%.1f", key="f_weight")
    friend_lthr = st.number_input("LTHR (bpm, optional if HR missing)", min_value=0, max_value=220, step=1, key="f_lthr")

alpha = st.slider("HR weighting exponent α", 0.5, 2.5, 1.5, 0.1)
smooth = st.slider("Smoothing seconds (EMA)", 1, 60, 30, 1)

compute_btn = st.button("⚡ Unleash nQDI™ Chaos")

# ---------------------------
# Compute
# ---------------------------
if compute_btn:
    if not your_file or not friend_file:
        st.error("Upload both files! We can't feed the nQDI™ apocalypse with missing data.")
        st.stop()

    try:
        df_you = load_file(your_file)
        df_friend = load_file(friend_file)
    except Exception as e:
        st.error(f"File read error: {e}")
        st.stop()

    # Check for missing HR
    if df_you['hr'].max() == 0 or df_friend['hr'].max() == 0:
        st.warning("⚠️ HR missing for one or both riders — nQDI™ will be computed purely from power where HR is absent. Accuracy reduced!")

    merged = align_to_same_length(df_you, df_friend)

    y_eff = effective_watts(merged['power_you'], merged['hr_you'], your_lthr, alpha)
    f_eff = effective_watts(merged['power_friend'], merged['hr_friend'], friend_lthr, alpha)

    y_eff_s = ema(y_eff, span=smooth)
    f_eff_s = ema(f_eff, span=smooth)

    nqdi_raw = f_eff_s / np.maximum(y_eff_s, 1e-6)
    nqdi_kg = (f_eff_s/friend_weight) / np.maximum(y_eff_s/your_weight,1e-6)
    nqdi_raw = np.clip(nqdi_raw, 0, 10)
    nqdi_kg = np.clip(nqdi_kg, 0, 10)

    st.metric("💀 nQDI™ Raw (mean)", f"{np.mean(nqdi_raw):.3f}")
    st.metric("⚖️ nQDI™ Weight-adjusted (mean)", f"{np.mean(nqdi_kg):.3f}")

    st.markdown("### 🔥 Brutal Roasts")
    st.info(roast_text(np.mean(nqdi_kg)))

    t = merged.index.to_numpy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Power & HR — witness your misery", "nQDI™ Live — humiliation in motion")
    )

    fig.add_trace(go.Scatter(x=t, y=merged['power_you'], name="You Power (watts of sorrow)", mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=merged['power_friend'], name="Friend Power (demonic efficiency)", mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=merged['hr_you'], name="You HR (panic bpm)", mode='lines', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=merged['hr_friend'], name="Friend HR (calm as hell bpm)", mode='lines', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=nqdi_raw, name="nQDI Raw (chaos)", mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=nqdi_kg, name="nQDI W/kg (brutality adjusted)", mode='lines'), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Break-even: mortal humiliation threshold", row=2, col=1)

    fig.update_yaxes(title_text="Power / HR", row=1, col=1)
    fig.update_yaxes(title_text="nQDI", row=2, col=1)
    fig.update_xaxes(title_text="Seconds")
    fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))

    st.plotly_chart(fig, use_container_width=True)
