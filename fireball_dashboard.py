import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
import json

def style_number(num, fireball=False):
    """Return HTML span for a styled circle number."""
    if fireball:
        return (f"<span style='display:inline-block; width:35px; height:35px; "
                f"border-radius:50%; background-color:orange; color:white; "
                f"text-align:center; line-height:35px; font-weight:bold; "
                f"margin:2px;'>{num}</span>")
    else:
        return (f"<span style='display:inline-block; width:35px; height:35px; "
                f"border-radius:50%; background-color:white; color:black; "
                f"text-align:center; line-height:35px; font-weight:bold; "
                f"margin:2px; border:1px solid black;'>{num}</span>")

st.set_page_config(page_title="Fireball Dashboard", layout="wide")
st.title("Fireball Dashboard")

# --- Google Sheets Setup ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"],
    scope
)
client = gspread.authorize(creds)

# Open Google Sheets
data_sheet = client.open("fireball_data").sheet1
rec_sheet = client.open("fireball_recommendations").sheet1

# Load draws as DataFrame
df = pd.DataFrame(data_sheet.get_all_records())
# Normalize columns and values
df.columns = df.columns.str.strip().str.lower()
if not df.empty:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()  # "Midday" / "Evening"
    df["fireball"] = df["fireball"].astype(str)
    # Ensure draw_sort exists BEFORE any sorting
    df["draw_sort"] = df["draw"].map({"Midday": 0, "Evening": 1})

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
# Decide which draw this recommendation is for (flip from the most recent logged draw)
if not df.empty:
    df_sorted = df.sort_values(["date", "draw_sort"], ascending=[True, True]).reset_index(drop=True)
    last_draw_row = df_sorted.iloc[-1]
    draw_type_for_rec = "Evening" if last_draw_row["draw"] == "Midday" else "Midday"
else:
    draw_type_for_rec = "Midday"  # fallback if no data yet

st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"🔥 Recommended Numbers for {draw_type_for_rec} Draw")
recent_window = df[pd.to_datetime(df["date"]) > (pd.to_datetime(df["date"]).max() - pd.Timedelta(days=14))] if not df.empty else df

# Fireball recommendation
if not df.empty and not recent_window.empty:
    recent_fire = recent_window["fireball"].value_counts(normalize=True)
    overall_fire = df["fireball"].value_counts(normalize=True)
    fire_combined = (recent_fire.add(overall_fire, fill_value=0)) / 2
    fire_rec = fire_combined.idxmax() if not fire_combined.empty else None
else:
    fire_rec = None

# Pick 3 recommendation
pick3 = []
if not df.empty and not recent_window.empty:
    for col in ["num1", "num2", "num3"]:
        recent_freq = recent_window[col].value_counts(normalize=True)
        overall_freq = df[col].value_counts(normalize=True)
        combined = (recent_freq.add(overall_freq, fill_value=0)) / 2
        pick3.append(str(combined.idxmax()) if not combined.empty else "0")
else:
    pick3 = ["0", "0", "0"]

if fire_rec:
    # Styled recommendation banner
    pick3_html = "".join([style_number(n) for n in pick3])
    fireball_html = style_number(fire_rec, fireball=True)
    st.markdown(
        f"<div style='background-color:#1f1c24; padding:15px; border-radius:10px; "
        f"text-align:center;'>"
        f"<div style='font-size:20px; font-weight:bold; color:white;'>"
        f"{pick3_html} + {fireball_html}</div></div>",
        unsafe_allow_html=True
    )

# --- Highlight Most Overdue Fireball ---
# Use same logic from streaks & gaps, but just pick the max gap
chron = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)
chron["pos"] = chron.index
last_pos = chron.groupby("fireball")["pos"].max()

N = len(chron)
gaps = {}
for d in [str(i) for i in range(10)]:
    if d in last_pos.index:
        gaps[d] = (N - 1) - int(last_pos.loc[d])
    else:
        gaps[d] = N  # never seen

# Find most overdue fireball
most_overdue = max(gaps, key=gaps.get)
gap_len = gaps[most_overdue]

# Styled overdue highlight
overdue_html = (
    f"<div style='font-size:16px; margin-top:10px;'>"
    f"⏳ Most Overdue Fireball: "
    f"<span style='display:inline-block; width:35px; height:35px; "
    f"border-radius:50%; background-color:gray; color:white; "
    f"text-align:center; line-height:35px; font-weight:bold; "
    f"margin-left:5px;'>{most_overdue}</span> "
    f"({gap_len} draws since last hit)</div>"
)

st.markdown(overdue_html, unsafe_allow_html=True)


# Log recommendation only once per draw
rec_data = rec_sheet.get_all_records()
today_str = str(datetime.now().date())
already_logged = any(
    str(row.get("date")) == today_str and str(row.get("draw")) == draw_type_for_rec
    for row in rec_data
    )
        if not already_logged:
            rec_sheet.append_row([today_str, draw_type_for_rec, ''.join(pick3), fire_rec])

# --- Quick View: Last 14 Draws ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🕒 Last 14 Draws (Pick 3 + Fireball)")

last14 = df.sort_values(["date", "draw_sort"], ascending=[False, True]).head(14)

styled_last14 = last14.copy()
styled_last14["Pick 3"] = styled_last14.apply(
    lambda r: "".join([style_number(r["num1"]), style_number(r["num2"]), style_number(r["num3"])]),
    axis=1
)
styled_last14["Fireball"] = styled_last14["fireball"].apply(lambda x: style_number(x, fireball=True))

# Convert to HTML without headers
styled_last14_html = styled_last14[["date", "draw", "Pick 3", "Fireball"]].to_html(
    escape=False, index=False, header=False
)

# Inject CSS: full width, centered text
styled_last14_html = styled_last14_html.replace(
    "<table border=\"1\" class=\"dataframe\">",
    "<table style='width:100%; border-collapse:collapse; font-size:16px; text-align:center;'>"
).replace(
    "<td>", "<td style='text-align:center; vertical-align:middle;'>"
).replace(
    "<th>", "<th style='text-align:center; vertical-align:middle;'>"
)

st.markdown(styled_last14_html, unsafe_allow_html=True)


# --- Frequency in Last 14 ---
if not df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Fireball Frequency (Last 14 Draws)")
    freq14 = last14["fireball"].value_counts().reindex([str(i) for i in range(10)], fill_value=0).reset_index()
    freq14.columns = ["Fireball", "Count"]
    fig0 = px.bar(freq14, x="Fireball", y="Count", text="Count", title="Frequency in Last 14 Draws")
    fig0.update_xaxes(type="category", categoryorder="array", categoryarray=[str(i) for i in range(10)])
    fig0.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    st.plotly_chart(fig0, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

# --- Streaks & Gaps ---
if not df.empty:
    st.subheader("⏳ Fireball Streaks & Gaps")
    chron = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)
    chron["pos"] = chron.index
    last_pos = chron.groupby("fireball")["pos"].max()
    digits = [str(i) for i in range(10)]
    N = len(chron)
    gaps = []
    for d in digits:
        gap = (N - 1) - int(last_pos.loc[d]) if d in last_pos.index else N
        gaps.append({"Fireball": d, "Draws Since Last Seen": gap})
    gaps_df = pd.DataFrame(gaps)
    fig_gaps = px.bar(gaps_df, x="Fireball", y="Draws Since Last Seen", text="Draws Since Last Seen",
                      title="How Long Since Each Fireball Last Hit")
    fig_gaps.update_xaxes(type="category", categoryorder="array", categoryarray=digits)
    fig_gaps.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    st.plotly_chart(fig_gaps, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

# --- Heatmap ---
if not df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Fireball by Weekday Heatmap")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
    heatmap_data = df.groupby(["weekday", "fireball"]).size().reset_index(name="count")
    fireball_order = [str(i) for i in range(10)]
    pivot = heatmap_data.pivot(index="weekday", columns="fireball", values="count") \
        .reindex(weekday_order).fillna(0)[fireball_order]
    fig3 = px.imshow(pivot, labels=dict(x="Fireball", y="Weekday", color="Count"),
                     x=fireball_order, y=weekday_order,
                     aspect="auto", color_continuous_scale="Viridis",
                     title="Fireball Frequency by Weekday")
    fig3.update_xaxes(tickmode="array", tickvals=list(range(10)), ticktext=[str(i) for i in range(10)], fixedrange=True)
    fig3.update_yaxes(fixedrange=True)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

# --- Recommendation History & Accuracy ---
rec_df = pd.DataFrame(rec_sheet.get_all_records())
if not rec_df.empty and not df.empty:
    rec_df.columns = rec_df.columns.str.strip().str.lower()
    rec_df["date"] = pd.to_datetime(rec_df["date"], errors="coerce").dt.date
    rec_df["recommended_pick3"] = rec_df["recommended_pick3"].apply(
        lambda x: ", ".join(list(str(x))) if pd.notna(x) else x
    )
    merged = pd.merge(df, rec_df, how="inner", left_on=["date", "draw"], right_on=["date", "draw"])
    if not merged.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Recommendation Accuracy History")
        merged["hit"] = merged.apply(
            lambda r: "✅" if str(r["fireball"]) == str(r["recommended_fireball"]) else "❌",
            axis=1
        )
        st.table(
            merged[["date", "draw", "recommended_pick3", "recommended_fireball", "fireball", "hit"]]
            .sort_values(["date", "draw"], ascending=[False, True])
            .head(20)
        )
        hit_rate = (merged["hit"] == "✅").mean() * 100
        st.write(f"Overall Fireball Hit Rate: **{hit_rate:.1f}%**")
        chart_df = merged.sort_values(["date", "draw"]).tail(30)
        chart_df["Hit Value"] = chart_df["hit"].map({"✅": 1, "❌": 0})
        fig_acc = px.scatter(chart_df, x="date", y="Hit Value", color="hit", symbol="draw",
                             title="Hit/Miss Over Time (Last 30 Draws)",
                             labels={"Hit Value": "Result", "date": "Date"},
                             color_discrete_map={"✅": "green", "❌": "red"})
        fig_acc.update_yaxes(tickvals=[0, 1], ticktext=["Miss", "Hit"], range=[-0.5, 1.5])
        st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
















