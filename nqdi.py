import streamlit as st
import pandas as pd
import numpy as np
import random

# -----------------------------------
# 🚴 nQDI™: The Quad Deficit Index
# -----------------------------------
st.title("🔥 nQDI™ — Quad Deficit Index Calculator 🔥")

st.markdown("""
Welcome to **nQDI™**, the only app that exists solely to measure how weak you are compared to your friends.  
Lower heart rate = better. Lower QDI = better. Higher QDI = cry yourself to sleep.  
If you don’t enter weight, don’t worry — we’ll still roast you mercilessly.
""")

# ----------------------------
# File upload
# ----------------------------
st.header("Upload your rides")
user_file = st.file_uploader("Upload your FIT/CSV file", type=["csv"])
friend_file = st.file_uploader("Upload your friend's FIT/CSV file", type=["csv"])

# ----------------------------
# Inputs
# ----------------------------
unit = st.selectbox("Weight unit", ["kg", "lbs"], key="weight_unit")

col1, col2 = st.columns(2)
with col1:
    user_weight = st.number_input("Your weight", min_value=0.0, value=0.0, step=0.1)
with col2:
    friend_weight = st.number_input("Friend's weight", min_value=0.0, value=0.0, step=0.1)

user_lthr = st.number_input("Your LTHR (optional)", min_value=0, value=0)
friend_lthr = st.number_input("Friend's LTHR (optional)", min_value=0, value=0)

# ----------------------------
# Helper functions
# ----------------------------
def load_data(file):
    df = pd.read_csv(file)
    if "speed" not in df or "power" not in df:
        st.error("File must contain at least 'speed' and 'power' columns.")
        return None
    # Remove actual stopped time
    df = df[df["speed"] > 0]
    return df

def normalized_power(power_series):
    rolling = power_series.rolling(window=30, min_periods=1).mean() ** 4
    return (rolling.mean()) ** 0.25

def calc_effort(df, lthr):
    npower = normalized_power(df["power"])
    if "heartrate" in df and lthr > 0 and df["heartrate"].mean() > 0:
        hr_ratio = df["heartrate"].mean() / lthr
    else:
        hr_ratio = 1  # fallback if no HR
    return npower / hr_ratio

def calc_qdi(user_effort, friend_effort, user_weight, friend_weight):
    raw_qdi = friend_effort / user_effort if user_effort > 0 else np.inf
    if user_weight > 0 and friend_weight > 0:
        adj_qdi = (friend_effort / friend_weight) / (user_effort / user_weight)
    else:
        adj_qdi = raw_qdi
    return raw_qdi, adj_qdi

# ----------------------------
# Roast generator
# ----------------------------
def roast(qdi):
    if qdi < 0.9:
        pool = [
            "Congrats, you’re the wattage cottage landlord now 🏠⚡",
            "Your quads are basically a renewable energy source at this point 🔋",
            "Friend better start riding an e-bike to keep up 🚲⚡"
        ]
    elif qdi < 1.1:
        pool = [
            "Neck and neck — a true Tuesday Night World Champs sprint finish 💀",
            "Barely surviving, like holding a wheel at 50kph in a crosswind 🌬️",
            "This is basically a photo finish KOM attempt 📸"
        ]
    elif qdi < 1.5:
        pool = [
            "Your friend is casually dismantling your soul like it’s Zwift warmup pace 💻",
            "QDI this high should be reported to the UCI for wattage doping 🚨",
            "Congrats, you’re the domestique in your own friendship group 🍾"
        ]
    else:
        pool = [
            "This isn’t QDI anymore, this is a public execution ⚔️",
            "Your legs called — they’ve officially filed for early retirement 🦵💀",
            "Friend didn’t just drop you, they erased you from Strava history 📉",
            "Your FTP now stands for *Friend Totally Pulverized* 🚴🔥"
        ]
    return random.choice(pool)

# ----------------------------
# Processing
# ----------------------------
if user_file and friend_file:
    user_df = load_data(user_file)
    friend_df = load_data(friend_file)

    if user_df is not None and friend_df is not None:
        user_effort = calc_effort(user_df, user_lthr)
        friend_effort = calc_effort(friend_df, friend_lthr)

        raw_qdi, adj_qdi = calc_qdi(user_effort, friend_effort, user_weight, friend_weight)

        st.subheader("Results")
        st.write(f"**Raw QDI:** {raw_qdi:.2f}")
        st.write(f"**Weight-adjusted QDI:** {adj_qdi:.2f}")

        st.markdown(f"### 🥵 Roast of Truth:\n{roast(adj_qdi)}")
