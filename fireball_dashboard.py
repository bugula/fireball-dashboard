import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
import json
import pytz


# ---------- styling helpers ----------
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

# ---------- Google Sheets setup ----------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], scope
)
client = gspread.authorize(creds)

# Open sheets
data_sheet = client.open("fireball_data").sheet1
rec_sheet  = client.open("fireball_recommendations").sheet1

# ---------- Load draws as DataFrame ----------
df = pd.DataFrame(data_sheet.get_all_records())
df.columns = df.columns.str.strip().str.lower()

if not df.empty:
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()   # "Midday" / "Evening"
    df["fireball"] = df["fireball"].astype(str)
    # sort helper: Evening above Midday when sorting desc by draw_sort
    df["draw_sort"] = df["draw"].map({"Midday": 0, "Evening": 1})

# ---------- Add New Drawing ----------
st.sidebar.header("➕ Add Latest Drawing")
with st.sidebar.form("new_draw_form"):
    est = pytz.timezone("US/Eastern")
    today_est = datetime.now(est).date()
    new_date = st.date_input("Draw Date", value=today_est)
    draw_type = st.selectbox("Draw Type", ["Midday", "Evening"])
    new_fireball = st.number_input("Fireball", 0, 9, step=1)
    num1 = st.number_input("Pick 3 - Number 1", 0, 9, step=1)
    num2 = st.number_input("Pick 3 - Number 2", 0, 9, step=1)
    num3 = st.number_input("Pick 3 - Number 3", 0, 9, step=1)
    submitted = st.form_submit_button("Add Drawing")

    if submitted:
        row = [str(new_date), draw_type, num1, num2, num3, new_fireball]
        data_sheet.append_row(row)
        st.sidebar.success(f"✅ Added {draw_type} draw {num1}{num2}{num3} + Fireball {new_fireball}")

# ======================================================================
#                           RECOMMENDATION ENGINE
#            (data-driven slot + locked to logged recommendation)
# ======================================================================

# 1) Decide which draw to recommend based on what's already logged in df
if not df.empty:
    last_date = df["date"].max()
    last_day  = df[df["date"] == last_date]
    draws_present = set(last_day["draw"].astype(str).str.strip().str.title())

    if "Midday" in draws_present and "Evening" in draws_present:
        # both done for last_date -> next is tomorrow's Midday
        rec_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).date()
        draw_type_for_rec = "Midday"
    elif "Midday" in draws_present:
        # only Midday entered -> next is same-day Evening
        rec_date = last_date
        draw_type_for_rec = "Evening"
    elif "Evening" in draws_present:
        # only Evening entered -> next is tomorrow's Midday
        rec_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).date()
        draw_type_for_rec = "Midday"
    else:
        # fallback (shouldn't happen normally)
        rec_date = last_date
        draw_type_for_rec = "Midday"
else:
    # no data yet -> start with today's Midday
    rec_date = datetime.now().date()
    draw_type_for_rec = "Midday"

st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"🔥 Recommended for {draw_type_for_rec} ({rec_date})")

# 2) Show the logged recommendation if it exists; otherwise compute + log + show
rec_data = rec_sheet.get_all_records()
rec_date_str = str(rec_date)

existing_rec = next(
    (row for row in rec_data
     if str(row.get("date")) == rec_date_str and
        str(row.get("draw")).strip().title() == draw_type_for_rec),
    None
)

if existing_rec:
    # ----- use logged rec (ensures banner matches sheet) -----
    pick3_str = str(existing_rec.get("recommended_pick3"))  # e.g. "123"
    fire_rec  = str(existing_rec.get("recommended_fireball"))

    pick3_html    = "".join([style_number(n) for n in pick3_str])
    fireball_html = style_number(fire_rec, fireball=True)

    st.markdown(
        f"<div style='background-color:#1f1c24; padding:15px; border-radius:10px; text-align:center;'>"
        f"<div style='font-size:20px; font-weight:bold; color:white;'>{pick3_html} + {fireball_html}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

else:
    # ----- compute a new recommendation -----
    if not df.empty:
        recent_window = df[pd.to_datetime(df["date"]) > (pd.to_datetime(df["date"]).max() - pd.Timedelta(days=14))]
    else:
        recent_window = df

    # Fireball
    fire_rec = None
    if not df.empty and not recent_window.empty:
        recent_fire   = recent_window["fireball"].value_counts(normalize=True)
        overall_fire  = df["fireball"].value_counts(normalize=True)
        fire_combined = (0.25 * recent_fire.add(0, fill_value=0)) + (0.75 * overall_fire)
        fire_rec = fire_combined.idxmax() if not fire_combined.empty else None

    # Pick 3 (slot-wise)
    if not df.empty and not recent_window.empty:
        pick3 = []
        for col in ["num1", "num2", "num3"]:
            recent_freq = recent_window[col].value_counts(normalize=True)
            overall_freq = df[col].value_counts(normalize=True)
            combined = (0.25 * recent_freq.add(0, fill_value=0)) + (0.75 * overall_freq)
            pick3.append(str(combined.idxmax()) if not combined.empty else "0")
    else:
        pick3 = ["0", "0", "0"]

    if fire_rec:
        pick3_html    = "".join([style_number(n) for n in pick3])
        fireball_html = style_number(fire_rec, fireball=True)

        st.markdown(
            f"<div style='background-color:#1f1c24; padding:15px; border-radius:10px; text-align:center;'>"
            f"<div style='font-size:20px; font-weight:bold; color:white;'>{pick3_html} + {fireball_html}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # ----- log once per (date, draw) -----
        rec_sheet.append_row([rec_date_str, draw_type_for_rec, ''.join(pick3), fire_rec])

# 3) Overdue highlight (based on full history)
if not df.empty:
    chron = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)
    chron["pos"] = chron.index
    last_pos = chron.groupby("fireball")["pos"].max()

    N = len(chron)
    gaps = {}
    for d in [str(i) for i in range(10)]:
        gaps[d] = (N - 1) - int(last_pos.loc[d]) if d in last_pos.index else N

    most_overdue = max(gaps, key=gaps.get)
    gap_len = gaps[most_overdue]

    overdue_html = (
        f"<div style='font-size:16px; margin-top:10px; text-align:center;'>"
        f"⏳ Overdue: "
        f"<span style='display:inline-block; width:35px; height:35px; border-radius:50%; "
        f"background-color:gray; color:white; text-align:center; line-height:35px; "
        f"font-weight:bold; margin-left:5px;'>{most_overdue}</span> "
        f"({gap_len} draws since last hit)</div>"
    )
    st.markdown(overdue_html, unsafe_allow_html=True)

# ======================================================================
#                           LAST 14 DRAWS (styled)
# ======================================================================
if not df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🕒 Last 14 Draws")

    last14 = df.sort_values(["date", "draw_sort"], ascending=[False, False]).head(14)

    styled_last14 = last14.copy()
    styled_last14["Pick 3"] = styled_last14.apply(
        lambda r: "".join([style_number(r["num1"]), style_number(r["num2"]), style_number(r["num3"])]),
        axis=1
    )
    styled_last14["Fireball"] = styled_last14["fireball"].apply(lambda x: style_number(x, fireball=True))

    # HTML table (no index)
    last14_html = styled_last14[["date", "draw", "Pick 3", "Fireball"]].to_html(
        escape=False, index=False, header=False
    )
    last14_html = last14_html.replace(
        "<table border=\"1\" class=\"dataframe\">",
        "<table style='width:100%; border-collapse:collapse; font-size:16px; text-align:center;'>"
    ).replace(
        "<td>", "<td style='text-align:center; vertical-align:middle;'>"
    ).replace(
        "<th>", "<th style='text-align:center; vertical-align:middle;'>"
    )
    st.markdown(last14_html, unsafe_allow_html=True)

# ======================================================================
#                       FIREBALL FREQUENCY (last 14)
# ======================================================================
if not df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Fireball Frequency")
    freq14 = (df.sort_values(["date", "draw_sort"], ascending=[False, False])
                .head(14)["fireball"]
                .value_counts()
                .reindex([str(i) for i in range(10)], fill_value=0)
                .reset_index())
    freq14.columns = ["Fireball", "Count"]
    fig0 = px.bar(freq14, x="Fireball", y="Count", text="Count", title="Last 14 Draws")
    fig0.update_xaxes(type="category", categoryorder="array", categoryarray=[str(i) for i in range(10)])
    fig0.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    st.plotly_chart(fig0, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

# ======================================================================
#                         STREAKS & GAPS
# ======================================================================
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

# ======================================================================
#                           HEATMAP
# ======================================================================
if not df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Fireball by Weekday Heatmap")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
    heatmap_data = df.groupby(["weekday", "fireball"]).size().reset_index(name="count")
    fireball_order = [str(i) for i in range(10)]
    pivot = (heatmap_data.pivot(index="weekday", columns="fireball", values="count")
             .reindex(weekday_order).fillna(0)[fireball_order])
    fig3 = px.imshow(pivot, labels=dict(x="Fireball", y="Weekday", color="Count"),
                     x=fireball_order, y=weekday_order,
                     aspect="auto", color_continuous_scale="Viridis",
                     title="Fireball Frequency by Weekday")
    fig3.update_xaxes(tickmode="array", tickvals=list(range(10)), ticktext=[str(i) for i in range(10)], fixedrange=True)
    fig3.update_yaxes(fixedrange=True)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

# ======================================================================
#                RECOMMENDATION HISTORY (Last 14 completed)
# ======================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📊 Last 14 Fireball Recommendations")

rec_df = pd.DataFrame(rec_sheet.get_all_records())

if not rec_df.empty and not df.empty:
    # Normalize
    rec_df.columns = rec_df.columns.str.strip().str.lower()
    df.columns     = df.columns.str.strip().str.lower()

    rec_df["date"] = pd.to_datetime(rec_df["date"], errors="coerce").dt.date
    rec_df["draw"] = rec_df["draw"].astype(str).str.strip().str.title()
    df["date"]     = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"]     = df["draw"].astype(str).str.strip().str.title()

    # Only completed draws (inner join)
    merged = pd.merge(
        rec_df,
        df[["date", "draw", "fireball"]],
        on=["date", "draw"],
        how="inner"
    )

    # Limit to last 14 completed
    merged = merged.sort_values(["date", "draw"], ascending=[False, False]).head(14)

    if not merged.empty:
        merged["hit"] = merged.apply(
            lambda r: "✅" if str(r["fireball"]) == str(r["recommended_fireball"]) else "❌",
            axis=1
        )

        # Hit rate vs baseline
        hit_rate = (merged["hit"] == "✅").mean() * 100
        perf_vs_baseline = hit_rate - 10
        perf_str = f"+{perf_vs_baseline:.1f}%" if perf_vs_baseline >= 0 else f"{perf_vs_baseline:.1f}%"
        st.write(f"Hit Rate: **{hit_rate:.1f}%** (vs baseline 10% → {perf_str})")

        # Evening above Midday within date
        merged["draw"] = pd.Categorical(merged["draw"], categories=["Evening", "Midday"], ordered=True)
        table_df = merged.sort_values(["date", "draw"], ascending=[False, True])[
            ["date", "draw", "recommended_fireball", "fireball", "hit"]
        ]

        # HTML table (no index)
        history_html = table_df.to_html(escape=False, index=False)
        history_html = history_html.replace(
            "<table border=\"1\" class=\"dataframe\">",
            "<table style='width:100%; border-collapse:collapse; font-size:16px; text-align:center;'>"
        ).replace(
            "<td>", "<td style='text-align:center; vertical-align:middle;'>"
        ).replace(
            "<th>", "<th style='text-align:center; vertical-align:middle;'>"
        )
        st.markdown(history_html, unsafe_allow_html=True)
    else:
        st.info("No completed recommendations to display yet.")
else:
    st.info("Not enough data to display recommendation accuracy.")

# ======================================================================
#                    ALL-TIME RECOMMENDATION ACCURACY
# ======================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📈 All-Time Recommendation Accuracy")

rec_df = pd.DataFrame(rec_sheet.get_all_records())

if not rec_df.empty and not df.empty:
    # Normalize
    rec_df.columns = rec_df.columns.str.strip().str.lower()
    df.columns     = df.columns.str.strip().str.lower()

    rec_df["date"] = pd.to_datetime(rec_df["date"], errors="coerce").dt.date
    rec_df["draw"] = rec_df["draw"].astype(str).str.strip().str.title()
    df["date"]     = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"]     = df["draw"].astype(str).str.strip().str.title()

    merged_all = pd.merge(
        rec_df,
        df[["date", "draw", "fireball"]],
        on=["date", "draw"],
        how="inner"   # only completed draws count
    )

    if not merged_all.empty:
        merged_all["hit"] = merged_all.apply(
            lambda r: "✅" if str(r["fireball"]) == str(r["recommended_fireball"]) else "❌",
            axis=1
        )
        hit_rate_all = (merged_all["hit"] == "✅").mean() * 100
        perf_vs_baseline = hit_rate_all - 10
        perf_str = f"+{perf_vs_baseline:.1f}%" if perf_vs_baseline >= 0 else f"{perf_vs_baseline:.1f}%"
        st.write(f"Hit Rate: **{hit_rate_all:.1f}%** (vs baseline 10% → {perf_str})")
    else:
        st.info("No completed recommendations to calculate all-time accuracy yet.")
else:
    st.info("Not enough data to display all-time accuracy.")



