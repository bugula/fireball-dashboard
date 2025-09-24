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

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

PICK3_URL = "https://www.illinoislottery.com/dbg/results/pick3"

# Selector assumptions (tune if the site tweaks markup):
# - A single “card” per draw row (the list on the first page).
# - Each card exposes date, draw label (Midday/Evening), pick-3 digits, and fireball.
# Inspect the page and adjust these selectors if needed.
CARD_SELECTOR        = "[data-testid='result-card'], .result-card, li[data-result]"  # fallback options
DATE_SELECTOR        = "[data-testid='draw-date'], .result-card__date, time"
DRAW_LABEL_SELECTOR  = "[data-testid='draw-name'], .result-card__draw"
PICK3_NUMS_SELECTOR  = "[data-testid='result-number'], .result-number, .result-card__numbers"
FIREBALL_SELECTOR    = "[data-testid='fireball'], .fireball, .result-card__fireball"

def wait_for_results(page, timeout=60000):
    # Wait until at least one result card shows up
    page.wait_for_selector(CARD_SELECTOR, timeout=timeout)

def accept_cookies_if_present(page):
    # Try a few common buttons; ignore errors
    for sel in [
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
        "button[aria-label*='accept' i]",
        "#onetrust-accept-btn-handler",
        "button:has-text('Got it')",
    ]:
        try:
            el = page.query_selector(sel)
            if el:
                el.click()
                page.wait_for_timeout(300)
                return
        except Exception:
            pass

def parse_card(card):
    # date
    date_text = (card.query_selector(DATE_SELECTOR).inner_text().strip()
                 if card.query_selector(DATE_SELECTOR) else "")
    # draw label
    draw_text = (card.query_selector(DRAW_LABEL_SELECTOR).inner_text().strip()
                 if card.query_selector(DRAW_LABEL_SELECTOR) else "")
    # pick3 numbers (join only digits)
    nums_text = ""
    if card.query_selector(PICK3_NUMS_SELECTOR):
        raw = card.query_selector(PICK3_NUMS_SELECTOR).inner_text()
        digits = [c for c in raw if c.isdigit()]
        # pad/trim to 3
        if len(digits) < 3: digits = (["0"] * (3 - len(digits))) + digits
        if len(digits) > 3: digits = digits[:3]
        nums_text = "".join(digits)

    # fireball (single digit)
    fireball = ""
    if card.query_selector(FIREBALL_SELECTOR):
        raw_fb = card.query_selector(FIREBALL_SELECTOR).inner_text()
        fb_digits = [c for c in raw_fb if c.isdigit()]
        fireball = fb_digits[0] if fb_digits else ""

    return {
        "date_text": date_text,
        "draw_text": draw_text,
        "pick3": nums_text,
        "fireball": fireball,
    }

def scrape_latest_cards(max_retries=2):
    """
    Returns a list of items like:
    [
      {"date_text": "...", "draw_text": "Midday", "pick3": "123", "fireball": "4"},
      ...
    ]
    """
    for attempt in range(1, max_retries + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                context = browser.new_context(
                    user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/124.0.0.0 Safari/537.36"),
                    locale="en-US",
                    timezone_id="America/Chicago",   # IL local time
                    viewport={"width": 1280, "height": 1600},
                )

                # Block heavy resources that aren't needed
                context.route("**/*", lambda route: (
                    route.abort()
                    if any(x in route.request.url.lower() for x in [
                        "doubleclick", "googletagmanager", "gtm.js",
                        "analytics", "hotjar", "facebook", "adservice",
                        ".mp4", ".webm", ".gif", ".svg", ".ttf", ".woff", ".woff2"
                    ])
                    else route.continue_()
                ))

                page = context.new_page()
                page.set_default_timeout(60000)  # 60s default per action

                # Go to the page; don't wait for networkidle (often never happens)
                page.goto(PICK3_URL, wait_until="domcontentloaded", timeout=60000)

                # Handle cookie banner if it pops
                accept_cookies_if_present(page)

                # Explicitly wait for the result cards
                wait_for_results(page, timeout=60000)

                # Grab cards
                cards = page.query_selector_all(CARD_SELECTOR)
                items = [parse_card(c) for c in cards]

                browser.close()

                # Basic sanity check
                if not items:
                    raise RuntimeError("No result cards found after selector wait.")

                return items

        except PWTimeout as e:
            if attempt == max_retries:
                raise
            # small backoff + retry
            # (site may be slow or first hit blocked by a challenge)
            # you can also add page.reload() in a single attempt if you prefer
        except Exception as e:
            if attempt == max_retries:
                raise
        # Wait a bit before retrying
        import time as _t
        _t.sleep(2)

    return []


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
