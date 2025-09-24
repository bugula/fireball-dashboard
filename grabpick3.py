# scrape_and_update.py
import os, json, re, time
from datetime import datetime
import pytz

from playwright.sync_api import sync_playwright
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

URL = "https://www.illinoislottery.com/dbg/results/pick3"

# --- Google Sheets auth ---
def get_gspread_client():
    # Expect the whole service-account JSON in env var GCP_SERVICE_ACCOUNT
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    if not creds_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT secret.")

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    return gspread.authorize(creds)

def open_sheets(gc):
    # Either open by name (both tabs named exactly as below) or by spreadsheet id + worksheet name
    # Names match what your app already uses:
    data_ws = gc.open("fireball_data").sheet1
    return data_ws

# --- Helpers: parsing ---
MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_RE = re.compile(rf"{MONTHS}\s+\d{{1,2}},\s+\d{{4}}", re.I)

def normalize_draw_type(s: str) -> str:
    s = (s or "").strip().lower()
    if "mid" in s: return "Midday"
    if "eve" in s: return "Evening"
    return s.title() if s else ""

def parse_text_block(txt: str):
    """
    Try to extract (date, draw, num1, num2, num3, fireball) from a chunk of text.
    We don’t rely on brittle CSS selectors; we scan text with regexes.
    """
    # Date in "September 24, 2025" style
    mdate = DATE_RE.search(txt)
    draw_date = mdate.group(0) if mdate else None

    # Draw type: look for Midday/Evening keywords
    draw = None
    if re.search(r"\bmid(day)?\b", txt, re.I):
        draw = "Midday"
    elif re.search(r"\beve(ning)?\b", txt, re.I):
        draw = "Evening"

    # The 3 main digits (allow spaces between)
    # Try to find 3 single digits near each other
    digits = re.findall(r"\b(\d)\b", txt)
    # Heuristic: Fireball appears with label; main three are often earlier in the block.
    # We’ll grab the first 3 distinct single-digit hits as num1-3.
    n1 = n2 = n3 = None
    if len(digits) >= 3:
        n1, n2, n3 = digits[0], digits[1], digits[2]

    # Fireball: look for "Fireball" label
    mfb = re.search(r"Fireball[:\s\-]*\s*(\d)", txt, re.I)
    fireball = mfb.group(1) if mfb else None

    if draw_date and draw and n1 is not None and fireball is not None:
        return {
            "date_str": draw_date,
            "draw": draw,
            "num1": int(n1),
            "num2": int(n2),
            "num3": int(n3),
            "fireball": fireball,
        }
    return None

def est_today():
    est = pytz.timezone("US/Eastern")
    return datetime.now(est).date()

def to_date(datestr: str):
    return datetime.strptime(datestr, "%B %d, %Y").date()

def already_in_sheet(ws, row):
    """Check if (date, draw) already present in the sheet."""
    # Read minimal columns to check
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return False
    # Normalize
    df.columns = df.columns.str.strip().str.lower()
    if not {"date", "draw"}.issubset(set(df.columns)):
        # assume empty/invalid sheet
        return False
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()
    want_date = row["date"]
    want_draw = row["draw"]
    return ((df["date"] == want_date) & (df["draw"] == want_draw)).any()

def upsert_latest(ws, items):
    """
    items: list of parsed dicts (each: date_str, draw, num1, num2, num3, fireball)
    We’ll convert to (date, draw) and append only if not present.
    """
    for it in items:
        try:
            d = to_date(it["date_str"])
        except Exception:
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
            ws.append_row([
                str(row_obj["date"]),
                row_obj["draw"],
                row_obj["num1"],
                row_obj["num2"],
                row_obj["num3"],
                row_obj["fireball"],
            ])
            print(f"Appended: {row_obj['date']} {row_obj['draw']} {row_obj['num1']}{row_obj['num2']}{row_obj['num3']} + FB {row_obj['fireball']}")
        else:
            print(f"Exists:   {row_obj['date']} {row_obj['draw']} — skipped")

def scrape_latest_cards():
    """
    Use Playwright to render the page and collect the top few result 'cards'.
    We’ll take the first ~4 logical blocks and parse them with regex.
    """
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.goto(URL, timeout=60000)
        # wait for any result containers; try a few generic selectors
        page.wait_for_load_state("networkidle")
        # give a little extra time for dynamic content (Cloudflare/JS)
        time.sleep(3)

        # Grab big text blobs from the page and split into blocks by double linebreaks
        text = page.inner_text("body")
        blocks = re.split(r"\n\s*\n", text)

        # Parse each block; keep first few plausible ones
        for b in blocks:
            parsed = parse_text_block(b)
            if parsed:
                results.append(parsed)
            if len(results) >= 6:  # enough (latest few entries)
                break

        browser.close()
    return results

def main():
    gc = get_gspread_client()
    ws = open_sheets(gc)
    items = scrape_latest_cards()
    if not items:
        print("No items parsed; nothing to upsert.")
        return

    # We’ll only attempt to upsert those from today or yesterday (guards against old noise)
    today = est_today()
    keep = []
    for it in items:
        try:
            d = to_date(it["date_str"])
        except Exception:
            continue
        if (today - d).days in (0, 1):  # today or yesterday
            keep.append(it)

    if not keep:
        print("Parsed items didn’t match today/yesterday; nothing to upsert.")
        return

    # Upsert into sheet
    upsert_latest(ws, keep)

if __name__ == "__main__":
    main()
