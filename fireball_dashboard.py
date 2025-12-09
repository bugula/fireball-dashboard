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
import time

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

# ---------- global page setup ----------
st.set_page_config(page_title="Fireball Dashboard", layout="wide")
st.title("Fireball Dashboard")

# ---------- global CSS (cards + bigger tabs) ----------
st.markdown("""
<style>
/* Reusable card style (matching Play Slate vibe) */
.fb-card {
  background:#1b1820;
  border:1px solid #2e2a34;
  border-radius:12px;
  padding:12px;
  margin-top:10px;
}
.fb-card .fb-card-title {
  font-weight:700;
  font-size:16px;
  color:#fff;
  margin-bottom:6px;
  display:flex;
  align-items:center;
  gap:8px;
}

/* Bigger, clearer tabs */
.stTabs [role="tablist"] {
  gap: 10px;
  border-bottom: 1px solid #2a2630;
  margin-top: 8px;
}
.stTabs [role="tab"] {
  padding: 10px 18px !important;
  border-radius: 10px 10px 0 0 !important;
  background: #1f1c24 !important;
  color: #e6e6e6 !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  border: 1px solid #2a2630 !important;
  border-bottom: none !important;
  opacity: 1 !important;
}
.stTabs [role="tab"][aria-selected="true"] {
  background: #2a263a !important;
  color: #ffffff !important;
  border-color: #3a3444 !important;
  border-bottom: none !important;
}
</style>
""", unsafe_allow_html=True)

def card_open(title_text: str):
    st.markdown(f"<div class='fb-card'><div class='fb-card-title'>{title_text}</div>", unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Google Sheets setup ----------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], scope
)
client = gspread.authorize(creds)

# Open sheets
data_sheet = client.open("fireball_data").sheet1
rec_sheet  = client.open("fireball_recommendations").sheet1

# ---------- NEW: model logs sheet ----------
MODEL_VERSION = "v1.1"

def open_log_sheet(client):
    """
    Use the 'model_logs' worksheet inside 'fireball_recommendations'.
    Assumes it already exists and has the correct headers.
    """
    try:
        host_ss = client.open("fireball_recommendations")
        ws = host_ss.worksheet("model_logs")
        return ws
    except Exception as e:
        st.warning(f"Could not open model_logs worksheet: {e}")
        return None

ws_logs = open_log_sheet(client)

def log_recommendation(ws_logs, rec_date, draw, hyper, top_combo, fire_rec, top_conf, slate_norm):
    # NOTE: kept for future use, but NOT called (to avoid duplicate writes).
    try:
        row = [
            time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            str(rec_date),
            draw,
            MODEL_VERSION,
            hyper.get("tau_days"), hyper.get("alpha"), hyper.get("uplift_clip"), hyper.get("sim_penalty"),
            top_combo, fire_rec, round(float(top_conf), 6) if top_conf is not None else "",
            json.dumps([{"p3": c, "conf": round(float(s), 6)} for c, s in slate_norm], separators=(',',':')),
            "", "", "", "", "", ""
        ]
        ws_logs.append_row(row)
    except Exception as e:
        st.warning(f"Logging failed: {e}")

def backfill_outcomes_and_scores(ws_logs, df_results):
    try:
        logs = ws_logs.get_all_records()
        if not logs:
            return
        logs_df = pd.DataFrame(logs)
        if logs_df.empty:
            return

        # Align types
        logs_df["date"] = pd.to_datetime(logs_df["date"], errors="coerce").dt.date
        df_results = df_results.copy()
        df_results["date"] = pd.to_datetime(df_results["date"], errors="coerce").dt.date
        df_results["draw"] = df_results["draw"].astype(str).str.title()

        m = pd.merge(
            logs_df,
            df_results[["date","draw","fireball"]],
            on=["date","draw"],
            how="left",
            suffixes=("","_real")
        )

        header = ws_logs.row_values(1)
        col = {name:i for i, name in enumerate(header)}  # 0-based

        for idx, r in m.iterrows():
            # only update rows that now have result and are not backfilled
            if pd.isna(r.get("fireball_real")):
                continue
            if str(r.get("realized_fireball") or "") != "":
                continue

            hit_top = 1 if str(r.get("top_fireball")) == str(r.get("fireball_real")) else 0
            # Use top_conf as proxy for p(hit_fb) (we don't have per-FB probs logged here)
            try:
                p_hit = float(r.get("top_conf"))
            except:
                p_hit = None

            if p_hit is None or p_hit < 0 or p_hit > 1:
                brier = ""
                logloss = ""
            else:
                brier = (p_hit - hit_top) ** 2
                eps = 1e-9
                logloss = -(hit_top * np.log(max(p_hit, eps)) + (1-hit_top)*np.log(max(1-p_hit, eps)))

            # read, modify, write the whole row in one go
            row_vals = ws_logs.row_values(idx + 2)  # +2 for header + 1-index
            if len(row_vals) < len(header):
                row_vals += [""]*(len(header)-len(row_vals))

            row_vals[col["realized_fireball"]] = str(r["fireball_real"])
            row_vals[col["hit_top"]]           = "1" if hit_top else "0"
            row_vals[col["brier_top"]]         = "" if brier == "" else f"{brier:.6f}"
            row_vals[col["logloss_top"]]       = "" if logloss == "" else f"{logloss:.6f}"
            # slate_hits kept empty here; can be computed if you later include FB in slate
            ws_logs.update(f"A{idx+2}", [row_vals])
    except Exception as e:
        st.warning(f"Backfill scoring failed: {e}")

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
        filt &= (pd.to_datetime(df_all["date"]).dt.day_name() == weekday)

    if filt.any():
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
    """
    if len(positions) == 0:
        return None, None, {}
    if len(positions) == 1:
        return None, None, {}

    gaps = [positions[i]-positions[i-1] for i in range(1, len(positions))]
    avg_gap = np.mean(gaps) if gaps else None

    from collections import Counter
    c = Counter(gaps)
    keys = sorted(c.keys())
    tail = {}
    running = 0
    for g in sorted(keys, reverse=True):
        running += c[g]
        tail[g] = running
    hazard = {g: c[g]/tail[g] for g in keys}
    return avg_gap, gaps[-1], hazard

def overdue_trigger(current_gap, hazard):
    if not hazard or current_gap is None:
        return False, None, None
    candidates = [g for g in hazard if g <= current_gap]
    if not candidates:
        return False, None, None
    g0 = max(candidates)
    h = float(hazard[g0])
    thr = float(np.percentile(list(hazard.values()), 75))
    return (h >= thr), h, thr

# ======================================================================
#        NEW: joint modeling + diversified 4-ticket slate helpers
# ======================================================================

def fireball_digit_uplift(df_all, fireball, slot_col):
    domain = [str(i) for i in range(10)]
    sub = df_all[df_all["fireball"].astype(str) == str(fireball)]
    if sub.empty:
        return pd.Series(1.0, index=domain)
    freq = sub[slot_col].astype(str).value_counts(normalize=True).reindex(domain, fill_value=0)
    uplift = (freq / 0.1).clip(0.75, 1.25)
    return uplift

def build_joint_candidates(df_all, rec_date, draw_type, top_per_slot=3, top_fireballs=3):
    if df_all.empty:
        return []

    rec_weekday = weekday_name(rec_date)
    chron_all = df_all.sort_values(["date","draw_sort"]).reset_index(drop=True)

    # Base probs
    pF_base = decayed_probs(chron_all["date"], chron_all["fireball"], tau_days=28, alpha=1.0)
    pF = conditional_uplift(chron_all, "fireball", pF_base, draw_type=draw_type, weekday=rec_weekday)

    p1 = decayed_probs(chron_all["date"], chron_all["num1"], tau_days=28, alpha=1.0)
    p2 = decayed_probs(chron_all["date"], chron_all["num2"], tau_days=28, alpha=1.0)
    p3 = decayed_probs(chron_all["date"], chron_all["num3"], tau_days=28, alpha=1.0)

    p1 = conditional_uplift(chron_all, "num1", p1, draw_type=draw_type, weekday=rec_weekday)
    p2 = conditional_uplift(chron_all, "num2", p2, draw_type=draw_type, weekday=rec_weekday)
    p3 = conditional_uplift(chron_all, "num3", p3, draw_type=draw_type, weekday=rec_weekday)

    idx1 = p1.sort_values(ascending=False).index[:top_per_slot]
    idx2 = p2.sort_values(ascending=False).index[:top_per_slot]
    idx3 = p3.sort_values(ascending=False).index[:top_per_slot]
    topF = pF.sort_values(ascending=False).index[:top_fireballs]

    uplift_1 = {fb: fireball_digit_uplift(chron_all, fb, "num1") for fb in topF}
    uplift_2 = {fb: fireball_digit_uplift(chron_all, fb, "num2") for fb in topF}
    uplift_3 = {fb: fireball_digit_uplift(chron_all, fb, "num3") for fb in topF}

    candidates = []
    for fb in topF:
        for a in idx1:
            for b in idx2:
                for c in idx3:
                    base = float(pF[fb] * p1[a] * p2[b] * p3[c])
                    co = float(uplift_1[fb][a] * uplift_2[fb][b] * uplift_3[fb][c])
                    score = base * co
                    candidates.append(("".join([a,b,c]), str(fb), score))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates

def combo_similarity(pickA, pickB):
    if pickA == pickB:
        return 999
    score = 0
    if pickA[-1] == pickB[-1]:
        score += 2
    sumA = sum(int(d) for d in pickA)
    sumB = sum(int(d) for d in pickB)
    if (sumA % 3) == (sumB % 3):
        score += 1
    return score

def select_diverse_slate(candidates, k=4, max_per_fireball=2, sim_penalty=0.15):
    slate = []
    count_by_fb = {}
    for pick, fb, raw_score in candidates:
        if len(slate) >= k:
            break
        if count_by_fb.get(fb, 0) >= max_per_fireball:
            continue
        penalty = 0.0
        for s_pick, s_fb, s_score in slate:
            penalty += combo_similarity(pick, s_pick) * sim_penalty
            if fb == s_fb:
                penalty += 0.1
        adj = raw_score * max(0.0, (1.0 - penalty))
        if adj <= 0:
            continue
        slate.append((pick, fb, adj))
        count_by_fb[fb] = count_by_fb.get(fb, 0) + 1

    if len(slate) < k:
        used = set((p, f) for p, f, _ in slate)
        for pick, fb, s in candidates:
            if len(slate) >= k:
                break
            if (pick, fb) in used:
                continue
            if count_by_fb.get(fb, 0) >= max_per_fireball:
                continue
            slate.append((pick, fb, s))
            count_by_fb[fb] = count_by_fb.get(fb, 0) + 1
    return slate[:k]

def render_four_ticket_slate(slate):
    if not slate:
        return
    total = sum(max(s, 0.0) for _, _, s in slate) or 1.0
    norm = [(p, f, s/total) for p, f, s in slate]

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Play Slate (4 tickets)")

    rows = []
    for pick, fb, w in norm:
        pc = f"{w*100:.0f}%"
        pick_html = "".join(style_number(d) for d in pick)
        fb_html   = style_number(fb, fireball=True)
        bar = f"<div style='height:6px; width:{max(6,int(w*100))}%; background:#3fb950; border-radius:4px;'></div>"
        rows.append(
            f"<div style='display:flex; align-items:center; justify-content:space-between; gap:16px; "
            f"background:#1f1c24; border:1px solid #2a2630; padding:10px 12px; border-radius:10px; margin:6px 0;'>"
            f"<div style='font-weight:700; color:#fff;'>{pick_html} + {fb_html}</div>"
            f"<div style='min-width:90px; text-align:right; color:#9fb5ff; font-weight:600;'>{pc}</div>"
            f"</div>"
            f"{bar}"
        )
    st.markdown("<div style='display:flex; flex-direction:column; gap:8px;'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

# ---------- Play Slate helpers (confidence bars for pick3-only view) ----------
def compute_display_slate(df_all, rec_date, draw_type_for_rec, top_per_slot=3, k=6):
    """Return [(combo_str, normalized_score_float), ...] ordered by strength desc."""
    if df_all.empty:
        return []
    rec_weekday = weekday_name(rec_date)
    chron_all = df_all.sort_values(["date", "draw_sort"]).reset_index(drop=True)

    p1 = decayed_probs(chron_all["date"], chron_all["num1"], tau_days=28, alpha=1.0, domain=DIGITS)
    p2 = decayed_probs(chron_all["date"], chron_all["num2"], tau_days=28, alpha=1.0, domain=DIGITS)
    p3 = decayed_probs(chron_all["date"], chron_all["num3"], tau_days=28, alpha=1.0, domain=DIGITS)

    p1 = conditional_uplift(chron_all, "num1", p1, draw_type=draw_type_for_rec, weekday=rec_weekday)
    p2 = conditional_uplift(chron_all, "num2", p2, draw_type=draw_type_for_rec, weekday=rec_weekday)
    p3 = conditional_uplift(chron_all, "num3", p3, draw_type=draw_type_for_rec, weekday=rec_weekday)

    combos = top_k_combos(p1, p2, p3, top_per_slot=3, k=k)
    total = sum(s for _, s in combos) or 1.0
    return [(c, s/total) for c, s in combos]

def _conf_bar_html(p: float) -> str:
    pct = max(0.0, min(1.0, float(p))) * 100.0
    return (
        "<div style='width:100%; height:10px; background:#2a2a2a; border-radius:999px;'>"
        f"  <div style='width:{pct:.0f}%; height:10px; background:#4aa3ff; border-radius:999px;'></div>"
        "</div>"
    )

def _combo_bubbles_html(combo: str) -> str:
    return "".join(style_number(ch) for ch in combo)

def render_play_slate(norm_list) -> str:
    if not norm_list:
        return ""
    rows = []
    for idx, (combo, score) in enumerate(norm_list, 1):
        rows.append(
            "<tr>"
            f"  <td style='text-align:center; padding:6px 8px; color:#bbb;'>{idx}</td>"
            f"  <td style='text-align:center; padding:6px 8px;'>{_combo_bubbles_html(combo)}</td>"
            f"  <td style='width:40%; padding:6px 8px;'>{_conf_bar_html(score)}</td>"
            f"  <td style='text-align:right; padding:6px 8px; color:#bbb;'>{score*100:.0f}%</td>"
            "</tr>"
        )
    table = (
        "<div class='fb-card'>"
        "<div class='fb-card-title'>🎯 Play Slate</div>"
        "<table style='width:100%; border-collapse:collapse;'>"
        "  <thead>"
        "    <tr>"
        "      <th style='text-align:center; color:#aaa; font-weight:600;'>#</th>"
        "      <th style='text-align:center; color:#aaa; font-weight:600;'>Combo</th>"
        "      <th style='text-align:left;   color:#aaa; font-weight:600;'>Confidence</th>"
        "      <th style='text-align:right;  color:#aaa; font-weight:600;'>Share</th>"
        "    </tr>"
        "  </thead>"
        f"  <tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )
    return table

# ---------- Health strip ----------
def render_health_strip(hit14, baseline=10.0, last_date=None, logloss60=None):
    perf = None if hit14 is None else (hit14 - baseline)
    perf_str = "" if perf is None else (f"{'+' if perf>=0 else ''}{perf:.1f}% vs {baseline:.0f}% base")
    fresh = f"Last draw: {last_date}" if last_date else ""
    logloss_str = "" if logloss60 is None else f" • LogLoss(60): {logloss60:.3f}"
    return (
        f"<div style='margin-top:8px; text-align:center; font-size:14px; color:#c8c8c8;'>"
        f"Health: <b>{'—' if hit14 is None else f'{hit14:.1f}%'} </b> {'' if perf_str=='' else perf_str}{logloss_str} • {fresh}"
        f"</div>"
    )

# --------- Safe, idempotent logging helper (logs once per date+draw) ---------
def log_model_once(ws_logs, key_date, key_draw, *, model_version, tau_days, alpha,
                   uplift_clip, sim_penalty, top_combo, fire_rec, top_conf, norm_list):
    """
    Append one row to model_logs exactly once per (date, draw).
    Uses both a sheet lookup and an in-session cache to suppress duplicates.
    """
    if ws_logs is None:
        return

    key_date = str(key_date).strip()
    key_draw = str(key_draw).strip().title()
    dedupe_key = f"{key_date}|{key_draw}"

    # per-session guard
    if "logged_keys" not in st.session_state:
        st.session_state.logged_keys = set()
    if dedupe_key in st.session_state.logged_keys:
        return

    # sheet-level guard
    try:
        rows = ws_logs.get_all_records()
        for r in rows:
            r_date = str(r.get("date", "")).strip()
            r_draw = str(r.get("draw", "")).strip().title()
            if r_date == key_date and r_draw == key_draw:
                st.session_state.logged_keys.add(dedupe_key)
                return
    except Exception:
        # If the read fails for some reason, fail-closed (don't write)
        return

    # write one row
    try:
        ws_logs.append_row([
            datetime.now(pytz.timezone("America/Chicago")).isoformat(),  # ts_logged
            key_date,                                                    # date
            key_draw,                                                    # draw
            model_version,                                               # model_version
            tau_days,                                                    # tau_days
            alpha,                                                       # alpha
            uplift_clip,                                                 # uplift_clip
            sim_penalty,                                                 # sim_penalty
            top_combo,                                                   # top_pick3
            fire_rec,                                                    # top_fireball
            round(float(top_conf), 6),                                   # top_conf
            json.dumps(norm_list),                                       # slate_json
            "", "",                                                      # realized_pick3, realized_fireball
            "", "",                                                      # hit_top, slate_hits
            "", ""                                                       # brier_top, logloss_top
        ])
        st.session_state.logged_keys.add(dedupe_key)
    except Exception:
        # swallow write errors (don’t risk cascading UI errors)
        pass

# ======================================================================
#                           RECOMMENDATION ENGINE
# ======================================================================

# 1) Decide which draw to recommend based on what's already logged in df
if not df.empty:
    last_date = df["date"].max()
    last_day  = df[df["date"] == last_date]
    draws_present = set(last_day["draw"].astype(str).str.strip().str.title())

    if "Midday" in draws_present and "Evening" in draws_present:
        rec_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).date()
        draw_type_for_rec = "Midday"
    elif "Midday" in draws_present:
        rec_date = last_date
        draw_type_for_rec = "Evening"
    elif "Evening" in draws_present:
        rec_date = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).date()
        draw_type_for_rec = "Midday"
    else:
        rec_date = last_date
        draw_type_for_rec = "Midday"
else:
    rec_date = datetime.now().date()
    draw_type_for_rec = "Midday"

st.markdown("<br>", unsafe_allow_html=True)
# Recommendation banner (kept as-is)
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

# hyperparams we log (surfaced if you want to tune later)
HYPER = {"tau_days": 28, "alpha": 1.0, "uplift_clip": "0.5-1.5", "sim_penalty": 0.15}

if existing_rec:
    # ----- use logged rec (ensures banner matches sheet) -----
    fire_rec  = str(existing_rec.get("recommended_fireball"))

    raw_pick3 = existing_rec.get("recommended_pick3")
    s = "" if raw_pick3 is None else str(raw_pick3).strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        digits = [d for d in parts if d.isdigit()]
    else:
        digits = [ch for ch in s if ch.isdigit()]
    if len(digits) < 3:
        digits = (["0"] * (3 - len(digits))) + digits
    elif len(digits) > 3:
        digits = digits[:3]
    top_combo = "".join(digits)

    pick3_html = "".join([style_number(n) for n in digits])
    fireball_html = style_number(fire_rec, fireball=True)

    # Banner card
    card_open("Recommendation")
    st.markdown(
        f"<div style='background-color:#1f1c24; padding:15px; border-radius:10px; text-align:center;'>"
        f"<div style='font-size:20px; font-weight:bold; color:white;'>{pick3_html} + {fireball_html}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    card_close()

    # --- Play Slate (confidence bars; display-only recompute) ---
    norm = []
    try:
        norm = compute_display_slate(df, rec_date, draw_type_for_rec, top_per_slot=3, k=6)
        if norm:
            st.markdown(render_play_slate(norm), unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not compute play slate: {e}")

    # --- TL;DR Health strip ---
    try:
        rec_df_health = pd.DataFrame(rec_sheet.get_all_records())
        if not rec_df_health.empty and not df.empty:
            rec_df_health.columns = rec_df_health.columns.str.strip().str.lower()
            rec_df_health["date"] = pd.to_datetime(rec_df_health["date"], errors="coerce").dt.date
            rec_df_health["draw"] = rec_df_health["draw"].astype(str).str.title()
            merged_h = pd.merge(rec_df_health, df[["date","draw","fireball"]], on=["date","draw"], how="inner")
            merged_h = merged_h.sort_values(["date","draw"], ascending=[False, False]).head(14)
            hit14 = float((merged_h["fireball"].astype(str) == merged_h["recommended_fireball"].astype(str)).mean()*100) if not merged_h.empty else None
        else:
            hit14 = None
        last_draw_date = df["date"].max() if not df.empty else None


        st.markdown(render_health_strip(hit14, last_date=last_draw_date), unsafe_allow_html=True)

    except Exception:
        pass

    # >>> IMPORTANT: NO LOGGING HERE (prevents per-refresh writes) <<<

else:
    # ----- compute a new recommendation (UPGRADED) -----
    if df.empty:
        st.info("Not enough data yet to compute a recommendation.")
    else:
        rec_weekday = weekday_name(rec_date)
        chron_all = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)

        # Fireball base (decayed + Bayesian)
        fire_probs_base = decayed_probs(
            series_dates=chron_all["date"],
            series_vals =chron_all["fireball"],
            tau_days=HYPER["tau_days"], alpha=HYPER["alpha"], domain=DIGITS
        )
        fire_probs = conditional_uplift(chron_all, "fireball", fire_probs_base,
                                        draw_type=draw_type_for_rec, weekday=rec_weekday)

        # Slot-wise base probs
        p1 = decayed_probs(chron_all["date"], chron_all["num1"], tau_days=HYPER["tau_days"], alpha=HYPER["alpha"], domain=DIGITS)
        p2 = decayed_probs(chron_all["date"], chron_all["num2"], tau_days=HYPER["tau_days"], alpha=HYPER["alpha"], domain=DIGITS)
        p3 = decayed_probs(chron_all["date"], chron_all["num3"], tau_days=HYPER["tau_days"], alpha=HYPER["alpha"], domain=DIGITS)

        p1 = conditional_uplift(chron_all, "num1", p1, draw_type=draw_type_for_rec, weekday=rec_weekday)
        p2 = conditional_uplift(chron_all, "num2", p2, draw_type=draw_type_for_rec, weekday=rec_weekday)
        p3 = conditional_uplift(chron_all, "num3", p3, draw_type=draw_type_for_rec, weekday=rec_weekday)

        # Top-K combos
        combos = top_k_combos(p1, p2, p3, top_per_slot=3, k=5)
        total_score = sum(s for _, s in combos) or 1.0
        norm = [(c, s/total_score) for c, s in combos]

        top_combo, top_combo_score = norm[0]
        fire_rec = fire_probs.sort_values(ascending=False).index[0]

        # Recommendation card
        card_open("Recommendation")
        pick3_html    = "".join([style_number(n) for n in list(top_combo)])
        fireball_html = style_number(fire_rec, fireball=True)
        st.markdown(
            f"<div style='background-color:#1f1c24; padding:15px; border-radius:10px; text-align:center;'>"
            f"<div style='font-size:20px; font-weight:bold; color:white;'>{pick3_html} + {fireball_html}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        card_close()

        # --- Play Slate (confidence bars; uses the same 'norm') ---
        try:
            if norm:
                st.markdown(render_play_slate(norm), unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not compute play slate: {e}")

        # --- TL;DR Health strip ---
        try:
            rec_df_health = pd.DataFrame(rec_sheet.get_all_records())
            if not rec_df_health.empty and not df.empty:
                rec_df_health.columns = rec_df_health.columns.str.strip().str.lower()
                rec_df_health["date"] = pd.to_datetime(rec_df_health["date"], errors="coerce").dt.date
                rec_df_health["draw"] = rec_df_health["draw"].astype(str).str.title()
                merged_h = pd.merge(rec_df_health, df[["date","draw","fireball"]], on=["date","draw"], how="inner")
                merged_h = merged_h.sort_values(["date","draw"], ascending=[False, False]).head(14)
                hit14 = float((merged_h["fireball"].astype(str) == merged_h["recommended_fireball"].astype(str)).mean()*100) if not merged_h.empty else None
            else:
                hit14 = None
            last_draw_date = df["date"].max() if not df.empty else None


            st.markdown(render_health_strip(hit14, last_date=last_draw_date), unsafe_allow_html=True)

        except Exception:
            pass

        # ----- log once per (date, draw) (ONLY HERE) -----
        log_model_once(
            ws_logs,
            rec_date_str,
            draw_type_for_rec,
            model_version=MODEL_VERSION,
            tau_days=HYPER["tau_days"],
            alpha=HYPER["alpha"],
            uplift_clip=HYPER["uplift_clip"],
            sim_penalty=HYPER["sim_penalty"],
            top_combo=top_combo,
            fire_rec=fire_rec,
            top_conf=top_combo_score,
            norm_list=norm,
        )

# ======================================================================
# Tabs to reduce clutter (now styled larger)
# ======================================================================
t_trends, t_gaps, t_diag = st.tabs(["Trends", "Gaps & Overdue", "Diagnostics"])

# ======================================================================
#                   TOP 2 OVERDUE NOW (chips) -> Gaps tab
# ======================================================================
with t_gaps:
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
            card_open("Overdue Now (Top 2)")
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
            chips_html = "<div style='text-align:center; margin-top:6px;'>" + " ".join(chips) + "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)
            card_close()

# ======================================================================
#                           LAST 14 DRAWS (styled) -> Trends
# ======================================================================
with t_trends:
    if not df.empty:
        card_open("🕒 Last 14 Draws")
        last14 = df.sort_values(["date", "draw_sort"], ascending=[False, False]).head(14)

        styled_last14 = last14.copy()
        styled_last14["Pick 3"] = styled_last14.apply(
            lambda r: "".join([style_number(r["num1"]), style_number(r["num2"]), style_number(r["num3"])]),
            axis=1
        )
        styled_last14["Fireball"] = styled_last14["fireball"].apply(lambda x: style_number(x, fireball=True))

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
        card_close()

# ======================================================================
#                       FIREBALL FREQUENCY (last 14) -> Trends
# ======================================================================
with t_trends:
    if not df.empty:
        card_open("📊 Fireball Frequency (Last 14)")
        freq14 = (df.sort_values(["date", "draw_sort"], ascending=[False, False])
                    .head(14)["fireball"]
                    .value_counts()
                    .reindex(DIGITS, fill_value=0)
                    .reset_index())
        freq14.columns = ["Fireball", "Count"]
        fig0 = px.bar(freq14, x="Fireball", y="Count", text="Count")  # no title here
        fig0.update_xaxes(type="category", categoryorder="array", categoryarray=DIGITS)
        fig0.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig0, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
        card_close()

# ======================================================================
#                         STREAKS & GAPS -> Gaps tab
# ======================================================================
with t_gaps:
    if not df.empty:
        card_open("⏳ Fireball Streaks & Gaps")
        chron = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)
        chron["pos"] = chron.index
        last_pos = chron.groupby("fireball")["pos"].max()
        N = len(chron)
        gaps = []
        for d in DIGITS:
            gap = (N - 1) - int(last_pos.loc[d]) if d in last_pos.index else N
            gaps.append({"Fireball": d, "Draws Since Last Seen": gap})
        gaps_df = pd.DataFrame(gaps)
        fig_gaps = px.bar(gaps_df, x="Fireball", y="Draws Since Last Seen", text="Draws Since Last Seen")
        fig_gaps.update_xaxes(type="category", categoryorder="array", categoryarray=DIGITS)
        fig_gaps.update_layout(xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
        st.plotly_chart(fig_gaps, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
        card_close()

# ======================================================================
#                         AVG CYCLES + HAZARD (fixed) -> Gaps tab
# ======================================================================
with t_gaps:
    if not df.empty:
        card_open("🔄 Fireball Cycle Analysis (Avg vs Current + Hazard)")
        chron = df.sort_values(["date", "draw_sort"]).reset_index(drop=True)
        chron["pos"] = chron.index

        results = []
        for d in DIGITS:
            positions = chron.index[chron["fireball"] == d].tolist()
            if len(positions) > 1:
                gap_lengths = [positions[i] - positions[i-1] for i in range(1, len(positions))]
                avg_gap = float(np.mean(gap_lengths)) if gap_lengths else None
                current_gap = len(chron) - 1 - positions[-1]
                overdue_pct = (current_gap / avg_gap) * 100 if avg_gap else None

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
                current_gap = len(chron)
                results.append({
                    "Fireball": d, "Avg Gap": None, "Current Gap": current_gap,
                    "Overdue %": None, "Hazard@Gap": None, "Hazard Thr(75%)": None, "Trigger?": "No"
                })

        gap_df = pd.DataFrame(results)
        if gap_df.empty:
            st.info("No cycle data yet.")
        else:
            gap_df["__sort_key"] = gap_df["Overdue %"].fillna(-1)
            gap_df = gap_df.sort_values(["__sort_key", "Current Gap"], ascending=[False, False]).drop(columns="__sort_key")

            # --- Highlighted HTML table inside card ---
            def row_html(row):
                base_cell = "text-align:center; color:#fff;"
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
                "<th style='color:#fff; text-align:center;'>Fireball</th>"
                "<th style='color:#fff; text-align:center;'>Avg Gap</th>"
                "<th style='color:#fff; text-align:center;'>Current Gap</th>"
                "<th style='color:#fff; text-align:center;'>Overdue %</th>"
                "<th style='color:#fff; text-align:center;'>Hazard@Gap</th>"
                "<th style='color:#fff; text-align:center;'>Hazard Thr(75%)</th>"
                "<th style='color:#fff; text-align:center;'>Trigger?</th>"
                "</tr></thead><tbody>"
            )
            try:
                body_html = "".join(row_html(r) for _, r in gap_df.iterrows())
                table_html = header_html + body_html + "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to render table: {e}")

            # Compact bar chart (no title)
            fig_gap_compare = px.bar(
                gap_df,
                x="Fireball", y=["Avg Gap", "Current Gap"],
                barmode="group"
            )
            fig_gap_compare.update_layout(
                xaxis=dict(tickmode="array", tickvals=DIGITS, ticktext=DIGITS, fixedrange=True),
                yaxis=dict(fixedrange=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(
                fig_gap_compare,
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False}
            )
        card_close()

# ======================================================================
#                           HEATMAP -> Trends
# ======================================================================
with t_trends:
    if not df.empty:
        card_open("Fireball by Weekday Heatmap")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
        heatmap_data = df.groupby(["weekday", "fireball"]).size().reset_index(name="count")
        fireball_order = DIGITS
        pivot = (heatmap_data.pivot(index="weekday", columns="fireball", values="count")
                 .reindex(weekday_order).fillna(0)[fireball_order])
        fig3 = px.imshow(
            pivot,
            labels=dict(x="Fireball", y="Weekday", color="Count"),
            x=fireball_order, y=weekday_order,
            aspect="auto", color_continuous_scale="Viridis"
        )  # no title
        fig3.update_xaxes(tickmode="array", tickvals=list(range(10)), ticktext=DIGITS, fixedrange=True)
        fig3.update_yaxes(fixedrange=True)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
        card_close()

# ======================================================================
#                RECOMMENDATION HISTORY (Last 14) -> Diagnostics
# ======================================================================
with t_diag:
    card_open("📊 Last 14 Fireball Recommendations")
    logs_df = pd.DataFrame(ws_logs.get_all_records())

    if not logs_df.empty and not df.empty:
        logs_df.columns = logs_df.columns.str.strip().str.lower()
        df.columns      = df.columns.str.strip().str.lower()

        logs_df["date"] = pd.to_datetime(logs_df["date"], errors="coerce").dt.date
        logs_df["draw"] = logs_df["draw"].astype(str).str.strip().str.title()
        df["date"]      = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["draw"]      = df["draw"].astype(str).str.strip().str.title()

        logs_df = logs_df.rename(columns={"top_fireball":"recommended_fireball"})

        merged = pd.merge(
            logs_df[["date","draw","recommended_fireball"]],
            df[["date","draw","fireball"]],
            on=["date","draw"],
            how="inner"
        ).sort_values(["date", "draw"], ascending=[False, False]).head(14)

        if not merged.empty:
            merged["hit"] = merged.apply(
                lambda r: "✅" if str(r["fireball"]) == str(r["recommended_fireball"]) else "❌",
                axis=1
            )

            hit_rate = (merged["hit"] == "✅").mean() * 100
            perf_vs_baseline = hit_rate - 10
            perf_str = f"+{perf_vs_baseline:.1f}%" if perf_vs_baseline >= 0 else f"{perf_vs_baseline:.1f}%"
            st.markdown(f"<div style='color:#c8c8c8;'>Hit Rate: <b>{hit_rate:.1f}%</b> (vs baseline 10% → {perf_str})</div>", unsafe_allow_html=True)

            merged["draw"] = pd.Categorical(merged["draw"], categories=["Evening", "Midday"], ordered=True)
            table_df = merged.sort_values(["date", "draw"], ascending=[False, True])[
                ["date", "draw", "recommended_fireball", "fireball", "hit"]
            ]

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
    card_close()

# ======================================================================
#                    ALL-TIME RECOMMENDATION ACCURACY -> Diagnostics
# ======================================================================
with t_diag:
    card_open("📈 All-Time Recommendation Accuracy")
    logs_df = pd.DataFrame(ws_logs.get_all_records())

    if not logs_df.empty and not df.empty:
        logs_df.columns = logs_df.columns.str.strip().str.lower()
        df.columns      = df.columns.str.strip().str.lower()

        logs_df["date"] = pd.to_datetime(logs_df["date"], errors="coerce").dt.date
        logs_df["draw"] = logs_df["draw"].astype(str).str.strip().str.title()
        df["date"]      = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["draw"]      = df["draw"].astype(str).str.strip().str.title()

        logs_df = logs_df.rename(columns={"top_fireball":"recommended_fireball"})

        merged = pd.merge(
            logs_df[["date","draw","recommended_fireball"]],
            df[["date","draw","fireball"]],
            on=["date","draw"],
            how="inner"
        )

        if not merged_all.empty:
            merged_all["hit"] = merged_all.apply(
                lambda r: "✅" if str(r["fireball"]) == str(r["recommended_fireball"]) else "❌",
                axis=1
            )
            hit_rate_all = (merged_all["hit"] == "✅").mean() * 100
            perf_vs_baseline = hit_rate_all - 10
            perf_str = f"+{perf_vs_baseline:.1f}%" if perf_vs_baseline >= 0 else f"{perf_vs_baseline:.1f}%"
            st.markdown(f"<div style='color:#c8c8c8;'>Hit Rate: <b>{hit_rate_all:.1f}%</b> (vs baseline 10% → {perf_str})</div>", unsafe_allow_html=True)
        else:
            st.info("No completed recommendations to calculate all-time accuracy yet.")
    else:
        st.info("Not enough data to display all-time accuracy.")
    card_close()

# ======================================================================
# Backfill outcomes → scores in logs
# ======================================================================
try:
    if not df.empty and ws_logs is not None:
        backfill_outcomes_and_scores(ws_logs, df)
except Exception as e:
    st.warning(f"Outcome backfill error: {e}")





