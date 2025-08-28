import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os

st.set_page_config(page_title="Fireball Dashboard", layout="wide")
st.title("Illinois Pick 3 + Fireball Dashboard")

# --- Google Sheets Setup ---
import json

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load creds from Streamlit Cloud Secrets
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    dict(st.secrets["gcp_service_account"]),
    scope
)
client = gspread.authorize(creds)


# Open Google Sheets
data_sheet = client.open("fireball_data").sheet1
rec_sheet = client.open("fireball_recommendations").sheet1

# Load draws as DataFrame
df = pd.DataFrame(data_sheet.get_all_records())
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df["fireball"] = df["fireball"].astype(str)

# --- Add New Drawing ---
st.sidebar.header("➕ Add Latest Drawing")
with st.sidebar.form("new_draw_form"):
    new_date = st.date_input("Draw Date")
    draw_type = st.selectbox("Draw Type", ["Midday", "Evening"])
    num1 = st.number_input("Number 1", 0, 9, step=1)
    num2 = st.number_input("Number 2", 0, 9, step=1)
    num3 = st.number_input("Number 3", 0, 9, step=1)
    new_fireball = st.selectbox("Fireball Number", [str(i) for i in range(10)])
    submitted = st.form_submit_button("Add Drawing")

    if submitted:
        row = [str(new_date), draw_type, num1, num2, num3, new_fireball]
        data_sheet.append_row(row)
        st.sidebar.success(f"✅ Added {draw_type} draw {num1}{num2}{num3} + Fireball {new_fireball}")

# --- Recommendation Engine ---
st.subheader("🔥 Recommended Numbers for NEXT Draw")
recent_window = df[pd.to_datetime(df["date"]) > (pd.to_datetime(df["date"]).max() - pd.Timedelta(days=14))]

# Fireball recommendation
recent_fire = recent_window["fireball"].value_counts(normalize=True)
overall_fire = df["fireball"].value_counts(normalize=True)
fire_combined = (recent_fire.add(overall_fire, fill_value=0)) / 2
fire_rec = fire_combined.idxmax() if not fire_combined.empty else None

# Pick 3 recommendation
pick3 = []
for col in ["num1", "num2", "num3"]:
    recent_freq = recent_window[col].value_counts(normalize=True)
    overall_freq = df[col].value_counts(normalize=True)
    combined = (recent_freq.add(overall_freq, fill_value=0)) / 2
    pick3.append(str(combined.idxmax()) if not combined.empty else "0")

if fire_rec:
    st.success(f"Recommended Next Draw: **{''.join(pick3)} + Fireball {fire_rec}**")

    # Log recommendation to rec_sheet
    rec_sheet.append_row([str(datetime.now().date()), ''.join(pick3), fire_rec])
    
# --- Quick View: Last 14 Draws ---
st.subheader("🕒 Last 14 Draws (Pick 3 + Fireball)")
df["draw_sort"] = df["draw"].map({"Midday": 0, "Evening": 1})
last14 = df.sort_values(["date", "draw_sort"], ascending=[False, True]).head(14)
last14["Pick 3"] = last14["num1"].astype(str) + ", " + last14["num2"].astype(str) + ", " + last14["num3"].astype(str)
st.table(last14[["date", "draw", "Pick 3", "fireball"]])

# --- Frequency in Last 14 ---
st.subheader("📊 Fireball Frequency (Last 14 Draws)")
freq14 = last14["fireball"].value_counts().reset_index()
freq14.columns = ["Fireball", "Count"]
fig0 = px.bar(freq14, x="Fireball", y="Count", text="Count", title="Frequency in Last 14 Draws")
st.plotly_chart(fig0, use_container_width=True)

# --- All Time Frequency ---
st.subheader("Fireball Frequency (All Time)")
freq = df["fireball"].value_counts().reset_index()
freq.columns = ["Fireball", "Count"]
fig1 = px.bar(freq, x="Fireball", y="Count", text="Count", title="Fireball Frequency")
st.plotly_chart(fig1, use_container_width=True)

# --- Heatmap by Weekday ---
st.subheader("Fireball by Weekday Heatmap")

# Create weekday column
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()

# Group data
heatmap_data = df.groupby(["weekday", "fireball"]).size().reset_index(name="count")

# Force fireball order 0–9
fireball_order = [str(i) for i in range(10)]

# Pivot for heatmap
pivot = heatmap_data.pivot(index="weekday", columns="fireball", values="count") \
    .reindex(weekday_order).fillna(0)[fireball_order]

# Plot heatmap
fig3 = px.imshow(
    pivot,
    labels=dict(x="Fireball", y="Weekday", color="Count"),
    x=fireball_order, y=weekday_order,
    aspect="auto", color_continuous_scale="Viridis",
    title="Fireball Frequency by Weekday"
)
st.plotly_chart(fig3, use_container_width=True)



