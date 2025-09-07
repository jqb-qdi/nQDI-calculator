# nQDI_v2.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fitparse import FitFile
import random

# ---------------------------
# CONFIG
# ---------------------------
titles = [
    "nQDI™: Pedal Bankruptcy Simulator",
    "nQDI™: Live Quadriceps Repossession Service",
    "nQDI™: Catastrophic Ego Foreclosure",
    "nQDI™: KOM Dreams in Ashes Edition",
    "nQDI™: Divorce-Grade Power Analysis"
]
st.set_page_config(
    page_title=random.choice(titles),
    layout="wide"
)

# ---------------------------
# UTILITIES
# ---------------------------
def read_fit(file_like):
    fitfile = FitFile(file_like)
    rows = []
    for msg in fitfile.get_messages("record"):
        rec = {"power": None, "hr": None, "speed": None}
        for d in msg:
            if d.name == "power": rec["power"] = d.value
            elif d.name == "heart_rate": rec["hr"] = d.value
            elif d.name == "speed": rec["speed"] = d.value
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["power"])
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    df["time"] = pd.to_timedelta(np.arange(len(df)), unit="s")
    return df[["time", "power", "hr", "speed"]]

def read_csv(file_like):
    df = pd.read_csv(file_like)
    if "speed" not in df.columns:
        df["speed"] = 0
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce").fillna(0)
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
    df["time"] = pd.to_timedelta(np.arange(len(df)), unit="s")
    return df[["time", "power", "hr", "speed"]]

def load_file(upload):
    name = upload.name.lower()
    if name.endswith(".fit"): return read_fit(io.BytesIO(upload.read()))
    elif name.endswith(".csv"): return read_csv(upload)
    else: raise ValueError("Unsupported file. Upload .fit or .csv")

def rolling_np(power, window=30):
    """ True Normalized Power calculation """
    p4 = pd.Series(power).rolling(window=window, min_periods=1).mean() ** 4
    return (np.mean(p4) ** 0.25)

def effective_power(power, hr, lthr, alpha=1.5):
    """ HR efficiency scaling: only if HR + LTHR are valid """
    if hr.max() <= 0 or lthr <= 0:
        return power
    hr_ratio = np.divide(lthr, np.maximum(hr, 1e-3))  # lower HR = better
    hr_ratio = np.clip(hr_ratio, 0.3, 3.0)
    return power * (hr_ratio ** alpha)

def align_to_same_length(a_df, b_df):
    min_len = min(len(a_df), len(b_df))
    return a_df.iloc[:min_len].reset_index(drop=True), b_df.iloc[:min_len].reset_index(drop=True)

def best_efforts(power, durations=[60, 300, 1200]):
    results = {}
    arr = np.array(power)
    for d in durations:
        if len(arr) >= d:
            rolling = pd.Series(arr).rolling(window=d).mean()
            results[f"{d//60}min"] = rolling.max()
        else:
            results[f"{d//60}min"] = np.nan
    return results

def roast_text(nqdi_kg):
    # Escalating tiers
    if nqdi_kg < 0.85:
        tier = [
            "🚀 Overlord status achieved. Your friend’s quads are crying for mercy.",
            "🦵💨 You basically made them your domestique for life."
        ]
    elif nqdi_kg < 1.0:
        tier = [
            "⚡ Barely tolerable: your friend’s watts still fear you.",
            "🥵 HR panic but hey, you’re not useless yet."
        ]
    elif nqdi_kg < 1.2:
        tier = [
            "💥 You’re trembling, their watts are flexing. Mediocrity unlocked.",
            "🤡 KOM attempts cancelled: friend casually spins past you."
        ]
    elif nqdi_kg < 1.5:
        tier = [
            "🔥 Welcome to the pain cave: population = your self-esteem.",
            "📉 Your FTP graph looks like a crypto crash chart."
        ]
    else:
        tier = [
            "💣 QUAD BANKRUPTCY. Retire now. Cry into your ✨handmade cotton tubulars✨.",
            "⚰️ Strava flagged your ride as ‘missing confidence.’"
        ]
    return random.choice(tier)

# ---------------------------
# UI
# ---------------------------
st.title("🩸 nQDI™: Catastrophic Quad Bankruptcy Deluxe Edition™")
st.markdown("This tool will roast you harder than a Cat 5 rider’s knees on a 20% climb. Upload your file, upload your friend’s file, and prepare for **ego destruction.**")

col1, col2 = st.columns(2)

with col1:
    st.subheader("You")
    your_file = st.file_uploader("Upload YOUR file", type=["fit","csv"], key="u_file")
    your_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, format="%.1f", key="u_weight")
    your_lthr = st.number_input("LTHR (bpm, optional)", min_value=0, max_value=250, step=1, key="u_lthr")

with col2:
    st.subheader("Friend")
    friend_file = st.file_uploader("Upload FRIEND file", type=["fit","csv"], key="f_file")
    friend_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, format="%.1f", key="f_weight")
    friend_lthr = st.number_input("LTHR (bpm, optional)", min_value=0, max_value=250, step=1, key="f_lthr")

alpha = st.slider("HR weighting exponent α", 0.5, 2.5, 1.5, 0.1)

compute_btn = st.button("⚡ Unleash nQDI™ Chaos")

# ---------------------------
# COMPUTE
# ---------------------------
if compute_btn:
    if not your_file or not friend_file:
        st.error("Upload both files, watt goblin. Missing data = missing roast.")
        st.stop()

    try:
        df_you = load_file(your_file)
        df_friend = load_file(friend_file)
    except Exception as e:
        st.error(f"File error: {e}")
        st.stop()

    # Remove stopped time (speed == 0)
    df_you = df_you[df_you["speed"] > 0].reset_index(drop=True)
    df_friend = df_friend[df_friend["speed"] > 0].reset_index(drop=True)

    # Align
    df_you, df_friend = align_to_same_length(df_you, df_friend)

    # Effective power
    y_eff = effective_power(df_you["power"].to_numpy(), df_you["hr"].to_numpy(), your_lthr, alpha)
    f_eff = effective_power(df_friend["power"].to_numpy(), df_friend["hr"].to_numpy(), friend_lthr, alpha)

    # Normalized Power
    you_np = rolling_np(y_eff)
    friend_np = rolling_np(f_eff)

    nqdi_raw = friend_np / max(you_np, 1e-6)
    nqdi_kg = (friend_np/friend_weight) / max(you_np/your_weight, 1e-6)

    st.metric("💀 nQDI™ Raw", f"{nqdi_raw:.3f}")
    st.metric("⚖️ nQDI™ W/kg", f"{nqdi_kg:.3f}")

    # Roasts
    st.markdown("### 🔥 Brutal Roast")
    st.info(roast_text(nqdi_kg))

    # Best Efforts
    st.markdown("### 📊 Best Efforts")
    efforts_you = best_efforts(y_eff)
    efforts_friend = best_efforts(f_eff)
    eff_table = pd.DataFrame({
        "You": efforts_you,
        "Friend": efforts_friend
    }).T
    st.table(eff_table)

    # Plot
    t = np.arange(len(y_eff))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Power + HR", "nQDI™ Over Time"))
    fig.add_trace(go.Scatter(x=t, y=df_you["power"], name="You Power"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_friend["power"], name="Friend Power"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_you["hr"], name="You HR", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df_friend["hr"], name="Friend HR", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=(f_eff/y_eff), name="nQDI Raw"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=(f_eff/friend_weight)/(y_eff/your_weight), name="nQDI W/kg"), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Break-even humiliation line", row=2, col=1)

    fig.update_yaxes(title_text="Watts / HR", row=1, col=1)
    fig.update_yaxes(title_text="nQDI", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_layout(height=700, hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)
