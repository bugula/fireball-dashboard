# scrape_and_update.py
import os, json, re, time
from datetime import datetime
import pytz
import requests
from bs4 import BeautifulSoup

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

BASE_URL  = "https://www.illinoislottery.com"
PICK3_URL = f"{BASE_URL}/dbg/results/pick3"

# ---------- Time helpers ----------
CT = pytz.timezone("America/Chicago")
def chicago_now():
    return datetime.now(CT)
def chicago_today():
    return chicago_now().date()

# ---------- Google Sheets ----------
def get_gspread_client():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    if not creds_json:
        b64 = os.environ.get("GCP_SERVICE_ACCOUNT_B64")
        if b64:
            import base64
            creds_json = base64.b64decode(b64).decode("utf-8")
    if not creds_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT (or GCP_SERVICE_ACCOUNT_B64).")

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    return gspread.authorize(creds)

def open_sheet(gc):
    sheet_id = os.environ.get("FIREBALL_DATA_SHEET_ID", "").strip()
    if sheet_id:
        ss = gc.open_by_key(sheet_id)
    else:
        ss = gc.open("fireball_data")
    ws = ss.sheet1
    print(f"[sheets] Opened spreadsheet: {ss.title} (sheet1: {ws.title})")
    return ws

# ---------- Parsing helpers ----------
MONTHS = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_RE = re.compile(rf"{MONTHS}\s+\d{{1,2}},\s+\d{{4}}", re.I)

def to_date(datestr: str):
    """Handle 'Sep 26, 2025' or 'September 26, 2025'."""
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(datestr.strip(), fmt).date()
        except ValueError:
            pass
    # Last resort: extract via regex then retry
    m = DATE_RE.search(datestr or "")
    if m:
        text = m.group(0)
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                pass
    raise ValueError(f"Unrecognized date format: {datestr!r}")

def normalize_draw_type(s: str) -> str:
    s = (s or "").strip().lower()
    if "mid" in s: return "Midday"
    if "eve" in s: return "Evening"
    return s.title() if s else ""

# ---------- Sheet dedupe ----------
def already_in_sheet(ws, row):
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return False
    df.columns = df.columns.str.strip().str.lower()
    if not {"date","draw"}.issubset(df.columns):
        return False
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()
    exists = ((df["date"] == row["date"]) & (df["draw"] == row["draw"])).any()
    if exists:
        print(f"[dup] Already have {row['date']} {row['draw']}")
    return exists

def upsert_latest(ws, items):
    appended = 0
    for it in items:
        try:
            d = to_date(it["date_str"])
        except Exception:
            print(f"[upsert] Bad date: {it.get('date_str')}, skipping")
            continue

        row_obj = {
            "date": d,
            "draw": it["draw"],
            "num1": it["num1"],
            "num2": it["num2"],
            "num3": it["num3"],
            "fireball": it["fireball"],
        }

        if not already_in_sheet(ws, row_obj):
            print(f"[append] -> {row_obj['date']} {row_obj['draw']} "
                  f"{row_obj['num1']}{row_obj['num2']}{row_obj['num3']} + FB {row_obj['fireball']}")
            ws.append_row([
                str(row_obj["date"]),
                row_obj["draw"],
                row_obj["num1"],
                row_obj["num2"],
                row_obj["num3"],
                row_obj["fireball"],
            ])
            appended += 1
        else:
            print(f"[append] skipped (exists) -> {row_obj['date']} {row_obj['draw']}")
    print(f"[upsert] Total appended: {appended}")
    return appended

# ---------- Scrape (list page only) ----------
def fetch_list_html(max_retries=3, timeout=30):
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
    for i in range(1, max_retries+1):
        try:
            r = requests.get(PICK3_URL, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"[fetch] Attempt {i} failed: {e}")
            if i == max_retries:
                raise
            time.sleep(1.5)

def parse_list_page(html: str):
    """
    Parse list-page cards like the snippet you shared (no detail navigation needed).
    Returns list of dicts:
      {date_str, draw, num1, num2, num3, fireball}
    """
    soup = BeautifulSoup(html, "lxml")
    items = []

    # Each card:
    # <li data-test-id="draw-result-1" ...>
    #   <span class="dbg-results__date-info">Sep 26, 2025</span>
    #   <span data-test-id="draw-result-schedule-type-text-1">midday</span>
    #   primary digits: div.grid-ball--pick3-primary--selected (3 nodes)
    #   secondary digit: div.grid-ball--pick3-secondary--selected (1 node)
    li_cards = soup.select('li[data-test-id^="draw-result-"]')
    print(f"[parse] Found {len(li_cards)} draw cards on list page")

    for li in li_cards:
        # date
        date_span = li.select_one(".dbg-results__date-info")
        date_str = (date_span.get_text(strip=True) if date_span else "").strip()

        # draw type
        draw_span = li.select_one('[data-test-id^="draw-result-schedule-type-text-"]')
        draw_raw  = (draw_span.get_text(strip=True) if draw_span else "").strip()
        draw = normalize_draw_type(draw_raw)

        # primary numbers (3)
        prim_nodes = li.select(".grid-ball--pick3-primary--selected")
        prim_digits = [n.get_text(strip=True) for n in prim_nodes if n.get_text(strip=True).isdigit()]
        prim_digits = prim_digits[:3]  # just in case page shows more

        # secondary (fireball) (1)
        sec_node = li.select_one(".grid-ball--pick3-secondary--selected")
        fb = (sec_node.get_text(strip=True) if sec_node else "").strip()

        if date_str and draw in ("Midday","Evening") and len(prim_digits) == 3 and fb.isdigit():
            item = {
                "date_str": date_str,             # e.g., "Sep 26, 2025"
                "draw": draw,                     # "Midday" or "Evening"
                "num1": int(prim_digits[0]),
                "num2": int(prim_digits[1]),
                "num3": int(prim_digits[2]),
                "fireball": fb,
            }
            items.append(item)
        else:
            print(f"[parse] Incomplete card skipped -> "
                  f"date={date_str!r}, draw={draw!r}, prim={prim_digits}, fb={fb!r}")

    return items

# ---------- MAIN ----------
def main():
    # 1) Sheets
    gc = get_gspread_client()
    ws = open_sheet(gc)

    # 2) Fetch & parse list page
    html = fetch_list_html()
    items = parse_list_page(html)
    if not items:
        print("No items parsed from list page; nothing to upsert.")
        return

    # 3) Keep only today / yesterday (Chicago)
    today = chicago_today()
    keep = []
    for it in items:
        try:
            d = to_date(it["date_str"])
        except Exception:
            print(f"[filter] Could not parse date_str: {it.get('date_str')!r}")
            continue
        delta = (today - d).days
        if delta in (0, 1):  # today or yesterday
            keep.append(it)
        print(f"[filter] {it['draw']} {it['date_str']} → {d} (delta={delta})")

    print(f"[filter] Rows kept for upsert: {len(keep)}")
    if not keep:
        print("Nothing within today/yesterday; aborting.")
        return

    # 4) Append if missing
    appended = upsert_latest(ws, keep)
    if appended == 0:
        print("[result] 0 rows appended (likely already present).")
    else:
        print(f"[result] Appended {appended} row(s).")

if __name__ == "__main__":
    main()
