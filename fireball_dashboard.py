import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import os
import json
import pytz
import numpy as np

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
    # ensure numeric digits as strings for slots
    for c in ["num1", "num2", "num3"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.extract(r"(\d)", expand=False).fillna("0")
    # sort helper: Evening above Midday when sorting desc by draw_sort
    df["draw_sort"] = df["draw"].map({"Midday": 0, "Evening": 1})

# ======================================================================
#                      UPGRADE HELPERS (new)
# ======================================================================

DIGITS = [str(i) for i in range(10)]
CT = pytz.timezone("America/Chicago")

def decayed_probs(series_dates, series_vals, tau_days=28, alpha=1.0, domain=DIGITS):
    """
    Exponentially-decayed relative frequencies with Dirichlet smoothing.
    series_dates: iterable of dates (date or datetime)
    series_vals:  iterable of digit strings ("0".."9")
    Returns pd.Series indexed by domain with probabilities.
    """
    if len(series_vals) == 0:
        return pd.Series([1/len(domain)]*len(domain), index=domain, dtype=float)

    sdates = pd.to_datetime(pd.Series(series_dates))
    svals  = pd.Series(series_vals).astype(str)
    today  = sdates.max()
    ages   = (today - sdates).dt.days.clip(lower=0)
    weights = np.exp(-ages / float(tau_days))

    dfw = pd.DataFrame({"d": svals, "w": weights})
    counts = dfw.groupby("d")["w"].sum()
    for k in domain:
        if k not in counts.index:
            counts.loc[k] = 0.0
    counts = counts.sort_index()

    probs = (counts + alpha) / (counts.sum() + alpha*len(domain))
    return probs.reindex(domain).astype(float)

def weekday_name(d):
    return pd.to_datetime(d).day_name()

def conditional_uplift(df_all, col, base_probs, draw_type=None, weekday=None, eps=1e-6):
    """
    Ratio p(x | conditions)/p(x) to nudge base_probs for context.
    col: "fireball" or "num1"/"num2"/"num3"
    """
    filt = pd.Series(True, index=df_all.index)
    if draw_type:
        filt &= (df_all["draw"] == draw_type)
    if weekday:
        # FIXED: use .dt.day_name() for Series
        filt &= (pd.to_datetime(df_all["date"]).dt.day_name() == weekday)

    if filt.any():
        # Use simple relative freq for uplift (already applying decay on base).
        sub = df_all.loc[filt, col].astype(str).value_counts()
        sub = sub.reindex(base_probs.index, fill_value=0).astype(float)
        sub = (sub + eps) / (sub.sum() + eps*len(base_probs))
        upl = (sub / (base_probs + eps)).clip(0.5, 1.5)  # guardrails
        adj = (base_probs * upl)
        adj = adj / adj.sum()
        return adj
    return base_probs

def top_k_combos(p1, p2, p3, top_per_slot=3, k=10):
    idx1 = p1.sort_values(ascending=False).index[:top_per_slot]
    idx2 = p2.sort_values(ascending=False).index[:top_per_slot]
    idx3 = p3.sort_values(ascending=False).index[:top_per_slot]
    rows = []
    for a in idx1:
        for b in idx2:
            for c in idx3:
                score = float(p1[a] * p2[b] * p3[c])
                rows.append(("".join([a,b,c]), score))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:k]

def gap_stats_and_hazard(positions):
    """
    positions: sorted list of indices where a fireball appeared (chronological).
    returns (avg_gap, current_gap, hazard_dict)
    hazard_dict: {gap: hazard} using empirical gaps
    """
    if len(positions) == 0:
        return None, None, {}
    if len(positions) == 1:
        return None, None, {}

    gaps = [positions[i]-positions[i-1] for i in range(1, len(positions))]
    avg_gap = np.mean(gaps) if gaps else None

    # empirical hazard h(g) = count(g) / count(gaps >= g)
    from collections import Counter
    c = Counter(gaps)
    keys = sorted(c.keys())
    tail = {}
    running = 0
    for g in sorted(keys, reverse=True):
        running += c[g]
        tail[g] = running
    hazard = {g: c[g]/tail[g] for g in keys}
    return avg_gap, gaps[-1], hazard  # current_gap since last success in gaps-space

def overdue_trigger(current_gap, hazard):
    if not hazard or current_gap is None:
        return False, None, None
    # nearest gap key <= current_gap
    candidates = [g for g in hazard if g <= current_gap]
    if not candidates:
        return False, None, None
    g0 = max(candidates)
    h = float(hazard[g0])
    thr = float(np.percentile(list(hazard.values()), 75))
    return (h >= thr), h, thr

# ---------- helper to ALWAYS produce alternates for display ----------
def compute_display_alternates(df_all, rec_date, draw_type_for_rec, top_per_slot=3, k=5):
    """Recompute combos for display-only (used even when a rec already exists)."""
    if df_all.empty:
        return []

    rec_weekday = weekday_name(rec_date)
    chron_all = df_all.sort_values(["date", "draw_sort"]).reset_index(drop=True)

    # slot-wise base probs
    p1 = decayed_probs(chron_all["date"], chron_all["num1"], tau_days=28, alpha=1.0, domain=DIGITS)
    p2 = decayed_probs(chron_all["date"], chron_all["num2"], tau_days=28, alpha=1.0, domain=DIGITS)
    p3 = decayed_probs(chron_all["date"], chron_all["num3"], tau_days=28, alpha=1.0, domain=DIGITS)

    # context uplift
    p1 = conditional_uplift(chron_all, "num1", p1, draw_type=draw_type_for_rec, weekday=rec_weekday)
    p2 = conditional_uplift(chron_all, "num2", p2, draw_type=draw_type_for_rec, weekday=rec_weekday)
    p3 = conditional_uplift(chron_all, "num3", p3, draw_type=draw_type_for_rec, weekday=rec_weekday)

    combos = top_k_combos(p1, p2, p3, top_per_slot=top_per_slot, k=k)
    total = sum(s for _, s in combos) or 1.0
    return [(c, s/total) for c, s in combos]

# ======================================================================
#                           RECOMMENDATION ENGINE
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
    (
        row for row in rec_data
        if str(row.get("date")) == rec_date_str
        and ((row.get("draw") or "").strip().title() == draw_type_for_rec)
    ),
    None
)

if existing_rec:
    # ----- use logged rec (ensures banner matches sheet) -----
    fire_rec  = str(existing_rec.get("recommended_fireball"))

    raw_pick3 = existing_rec.get("recommended_pick3")
    # Robust parse: handle "1,2,3", 123, "065", or even "65"
    s = "" if raw_pick3 is None else str(raw_pick3).strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        digits = [d for d in parts if d.isdigit()]
    else:
        digits = [ch for ch in s if ch.isdigit()]
    # Ensure exactly 3 digits (preserve leading zeros)
    if len(digits) < 3:
        digits = (["0"] * (3 - len(digits))) + digits
    elif len(digits) > 3:
        digits = digits[:3]

    pick3_html = "".join([style_number(n) for n in digits])
    fireball_html = style_number(fire_rec, fireball=True)

    st.markdown(
        f"<div style='background-color:#1f1c24; padding:15px; border-radius:10px; text-align:center;'>"
        f"<div style='font-size:20px; font-weight:bold; color:white;'>{pick3_html} + {fireball_html}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # --- ALWAYS show alternates (display-only recompute) ---
    norm = compute_display_alternates(df, rec_date, draw_type_for_rec, top_per_slot=3, k=5)
    if len(norm) > 1:
        chips = []
        for combo, s in norm[:3]:
            pct = f"{s*100:.0f}%"
            chip = (
                f"<span style='display:inline-flex; align-items:center; gap:6px; "
                f"background:#e8f3ff; border:1px solid #bcdcff; border-radius:999px; "
                f"padding:4px 10px; margin:4px;'>"
                f"{''.join(style_number(n) for n in combo)}"
                f"<span style='font-weight:600; color:#0b4a8b;'> {pct}</span>"
                f"</span>"
            )
            chips.append(chip)
        st.markdown("<div style='text-align:center; margin-top:6px;'>Alternates: " + " ".join(chips) + "</div>", unsafe_allow_html=True)

else:
    # ----- compute a new recommendation (UPGRADED) -----
    if df.empty:
        st.info("Not enough data yet to compute a recommendation.")
    else:
        # Context for uplift
        rec_weekday = weekday_name(rec_date)

        chron_all = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)

        # Fireball base (decayed + Bayesian)
        fire_probs_base = decayed_probs(
            series_dates=chron_all["date"],
            series_vals =chron_all["fireball"],
            tau_days=28, alpha=1.0, domain=DIGITS
        )
        # Apply uplift for draw type and weekday
        fire_probs = conditional_uplift(chron_all, "fireball", fire_probs_base,
                                        draw_type=draw_type_for_rec, weekday=rec_weekday)

        # Slot-wise base probs
        p1 = decayed_probs(chron_all["date"], chron_all["num1"], tau_days=28, alpha=1.0, domain=DIGITS)
        p2 = decayed_probs(chron_all["date"], chron_all["num2"], tau_days=28, alpha=1.0, domain=DIGITS)
        p3 = decayed_probs(chron_all["date"], chron_all["num3"], tau_days=28, alpha=1.0, domain=DIGITS)

        # Apply the same uplift to slots (from historical correlations)
        p1 = conditional_uplift(chron_all, "num1", p1, draw_type=draw_type_for_rec, weekday=rec_weekday)
        p2 = conditional_uplift(chron_all, "num2", p2, draw_type=draw_type_for_rec, weekday=rec_weekday)
        p3 = conditional_uplift(chron_all, "num3", p3, draw_type=draw_type_for_rec, weekday=rec_weekday)

        # Top-K combos
        combos = top_k_combos(p1, p2, p3, top_per_slot=3, k=5)
        # Normalize combo scores for a simple "confidence bar"
        total_score = sum(s for _, s in combos) or 1.0
        norm = [(c, s/total_score) for c, s in combos]

        # Choose top1 for logging (keep your rec_sheet schema)
        top_combo, top_combo_score = norm[0]
        fire_rec = fire_probs.sort_values(ascending=False).index[0]

        # Banner
        pick3_html    = "".join([style_number(n) for n in list(top_combo)])
        fireball_html = style_number(fire_rec, fireball=True)
        st.markdown(
            f"<div style='background-color:#1f1c24; padding:15px; border-radius:10px; text-align:center;'>"
            f"<div style='font-size:20px; font-weight:bold; color:white;'>{pick3_html} + {fireball_html}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Small list of alternates (now guaranteed to show)
        if len(norm) > 1:
            chips = []
            for combo, s in norm[:3]:
                pct = f"{s*100:.0f}%"
                chip = (
                    f"<span style='display:inline-flex; align-items:center; gap:6px; "
                    f"background:#e8f3ff; border:1px solid #bcdcff; border-radius:999px; "
                    f"padding:4px 10px; margin:4px;'>"
                    f"{''.join(style_number(n) for n in combo)}"
                    f"<span style='font-weight:600; color:#0b4a8b;'> {pct}</span>"
                    f"</span>"
                )
                chips.append(chip)
            st.markdown("<div style='text-align:center; margin-top:6px;'>Alternates: " + " ".join(chips) + "</div>", unsafe_allow_html=True)

        # ----- log once per (date, draw) -----
        rec_sheet.append_row([rec_date_str, draw_type_for_rec, top_combo, fire_rec])

# ======================================================================
#                   TOP 2 OVERDUE NOW (chips under banner)
# ======================================================================
if not df.empty:
    chron_all = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)
    chron_all["pos"] = chron_all.index
    results_for_rank = []
    for d in DIGITS:
        positions = chron_all.index[chron_all["fireball"] == d].tolist()
        if len(positions) > 1:
            gaps = [positions[i] - positions[i-1]] if len(positions) == 2 else [positions[i] - positions[i-1] for i in range(1, len(positions))]
            avg_gap = sum(gaps) / len(gaps)
            current_gap = (len(chron_all) - 1) - positions[-1]
            if avg_gap > 0:
                ratio = current_gap / avg_gap
                results_for_rank.append((d, current_gap, avg_gap, ratio))
        elif len(positions) == 1:
            current_gap = (len(chron_all) - 1) - positions[-1]
            results_for_rank.append((d, current_gap, None, -1))

    top2 = [t for t in sorted(results_for_rank, key=lambda x: x[3], reverse=True) if t[2] is not None][:2]

    if top2:
        chips = []
        for digit, cur_gap, avg_gap, ratio in top2:
            pct = f"{ratio*100:.0f}%"
            chip = (
                f"<span style='display:inline-flex; align-items:center; gap:6px; "
                f"background:#fff6c2; border:1px solid #f1de85; border-radius:999px; "
                f"padding:4px 10px; margin:4px;'>"
                f"{style_number(digit)}"
                f"<span style='font-weight:600; color:#5a4a00;'> {pct} of avg • gap {cur_gap}</span>"
                f"</span>"
            )
            chips.append(chip)
        chips_html = "<div style='text-align:center; margin-top:6px;'>Overdue now: " + " ".join(chips) + "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

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
                .reindex(DIGITS, fill_value=0)
                .reset_index())
    freq14.columns = ["Fireball", "Count"]
    fig0 = px.bar(freq14, x="Fireball", y="Count", text="Count", title="Last 14 Draws")
    fig0.update_xaxes(type="category", categoryorder="array", categoryarray=DIGITS)
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
    N = len(chron)
    gaps = []
    for d in DIGITS:
        gap = (N - 1) - int(last_pos.loc[d]) if d in last_pos.index else N
        gaps.append({"Fireball": d, "Draws Since Last Seen": gap})
    gaps_df = pd.DataFrame(gaps)
    fig_gaps = px.bar(gaps_df, x="Fireball", y="Draws Since Last Seen", text="Draws Since Last Seen",
                      title="How Long Since Each Fireball Last Hit")
    fig_gaps.update_xaxes(type="category", categoryorder="array", categoryarray=DIGITS)
    fig_gaps.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    st.plotly_chart(fig_gaps, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

# ======================================================================
#                         AVG CYCLES + HAZARD (new)
# ======================================================================
if not df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔄 Fireball Cycle Analysis (Avg vs Current + Hazard)")

    chron = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)
    chron["pos"] = chron.index
    N = len(chron)

    results = []
    for d in DIGITS:
        positions = chron.index[chron["fireball"] == d].tolist()
        if len(positions) > 1:
            # avg gap, "current gap" = distance since last hit in draw counts
            gap_lengths = [positions[i] - positions[i-1] for i in range(1, len(positions))]
            avg_gap = float(np.mean(gap_lengths)) if gap_lengths else None
            current_gap = len(chron) - 1 - positions[-1]
            overdue_pct = (current_gap / avg_gap) * 100 if avg_gap else None

            # hazard based on empirical gaps
            _, last_gap_in_gaps, hazard = gap_stats_and_hazard(positions)
            trig, hz, thr = overdue_trigger(last_gap_in_gaps, hazard)
            results.append({
                "Fireball": d,
                "Avg Gap": round(avg_gap, 1) if avg_gap else None,
                "Current Gap": current_gap,
                "Overdue %": round(overdue_pct, 0) if overdue_pct is not None else None,
                "Hazard@Gap": None if hz is None else round(hz, 3),
                "Hazard Thr(75%)": None if thr is None else round(thr, 3),
                "Trigger?": "Yes" if trig else "No"
            })
        else:
            current_gap = len(chron)  # never/once seen
            results.append({"Fireball": d, "Avg Gap": None, "Current Gap": current_gap,
                            "Overdue %": None, "Hazard@Gap": None, "Hazard Thr(75%)": None, "Trigger?": "No"})

    gap_df = pd.DataFrame(results)
    gap_df["__sort_key"] = gap_df["Overdue %"].fillna(-1)
    gap_df = gap_df.sort_values(["__sort_key", "Current Gap"], ascending=[False, False]).drop(columns="__sort_key")

# Build highlighted HTML table
def row_html(row):
    # default (non-highlighted) rows: white text
    base_cell = "text-align:center; color:#fff;"
    # highlighted rows: darker text for contrast on the yellow
    hi_row_style  = "background:#ffd36b;"
    hi_cell_style = "text-align:center; color:#111; font-weight:700;"

    if row.get("Trigger?") == "Yes":
        row_style = hi_row_style
        cell_style = hi_cell_style
    else:
        row_style = ""
        cell_style = base_cell

    return (
        f"<tr style='{row_style}'>"
        f"<td style='{cell_style}'>{row['Fireball']}</td>"
        f"<td style='{cell_style}'>{'' if pd.isna(row['Avg Gap']) else row['Avg Gap']}</td>"
        f"<td style='{cell_style}'>{row['Current Gap']}</td>"
        f"<td style='{cell_style}'>{'' if pd.isna(row['Overdue %']) else int(row['Overdue %'])}%</td>"
        f"<td style='{cell_style}'>{'' if pd.isna(row['Hazard@Gap']) else row['Hazard@Gap']}</td>"
        f"<td style='{cell_style}'>{'' if pd.isna(row['Hazard Thr(75%)']) else row['Hazard Thr(75%)']}</td>"
        f"<td style='{cell_style}'>{row['Trigger?']}</td>"
        "</tr>"
    )

    header_html = (
        "<table style='width:100%; border-collapse:collapse; font-size:16px;'>"
        "<thead><tr>"
        "<th style='color:#fff;'>Fireball</th>"
        "<th style='color:#fff;'>Avg Gap</th>"
        "<th style='color:#fff;'>Current Gap</th>"
        "<th style='color:#fff;'>Overdue %</th>"
        "<th style='color:#fff;'>Hazard@Gap</th>"
        "<th style='color:#fff;'>Hazard Thr(75%)</th>"
        "<th style='color:#fff;'>Trigger?</th>"
        "</tr></thead><tbody>"
    )
    body_html = "".join(row_html(r) for _, r in gap_df.iterrows())
    table_html = header_html + body_html + "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # Optional chart (kept; follows the sorted data)
    fig_gap_compare = px.bar(
        gap_df,
        x="Fireball", y=["Avg Gap", "Current Gap"],
        barmode="group",
        title="Average vs Current Gaps by Fireball"
    )
    fig_gap_compare.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=DIGITS,
            ticktext=DIGITS,
            fixedrange=True
        ),
        yaxis=dict(fixedrange=True)
    )
    st.plotly_chart(
        fig_gap_compare,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": False}
    )

# ======================================================================
#                           HEATMAP
# ======================================================================
if not df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Fireball by Weekday Heatmap")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
    heatmap_data = df.groupby(["weekday", "fireball"]).size().reset_index(name="count")
    fireball_order = DIGITS
    pivot = (heatmap_data.pivot(index="weekday", columns="fireball", values="count")
             .reindex(weekday_order).fillna(0)[fireball_order])
    fig3 = px.imshow(pivot, labels=dict(x="Fireball", y="Weekday", color="Count"),
                     x=fireball_order, y=weekday_order,
                     aspect="auto", color_continuous_scale="Viridis",
                     title="Fireball Frequency by Weekday")
    fig3.update_xaxes(tickmode="array", tickvals=list(range(10)), ticktext=DIGITS, fixedrange=True)
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


