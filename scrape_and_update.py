# scrape_and_update.py
import os, json, re, time, traceback
from datetime import datetime
import pytz

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PICK3_URL = "https://www.illinoislottery.com/dbg/results/pick3"

# Broad selector candidates (site markup can change)
CARD_SELECTOR_CANDIDATES = [
    "[data-testid='result-card']",
    ".result-card",
    "li[data-result]",
    "li:has([data-testid='result-number'])",
    "article:has([data-testid='result-number'])",
    "section:has-text('Pick 3') li",
    "section:has-text('Pick 3') article",
]

DATE_CANDIDATES = [
    "[data-testid='draw-date']",
    ".result-card__date",
    "time[datetime]",
    "time",
]

DRAW_LABEL_CANDIDATES = [
    "[data-testid='draw-name']",
    ".result-card__draw",
    "span:has-text('Midday')",
    "span:has-text('Evening')",
    "*:text('Midday')",
    "*:text('Evening')",
]

PICK3_NUMS_CANDIDATES = [
    "[data-testid='result-number']",
    ".result-number",
    ".result-card__numbers",
    "[class*='number']",
]

FIREBALL_CANDIDATES = [
    "[data-testid='fireball']",
    ".fireball",
    ".result-card__fireball",
    "span:has-text('Fireball')",
    "div:has-text('Fireball')",
]

# ---------------------------------------------------------------------
# Google Sheets auth
# ---------------------------------------------------------------------
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
    # Open by title; matches your app
    data_ws = gc.open("fireball_data").sheet1
    return data_ws

# ---------------------------------------------------------------------
# Helpers: parsing & dates
# ---------------------------------------------------------------------
MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_RE = re.compile(rf"{MONTHS}\s+\d{{1,2}},\s+\d{{4}}", re.I)

def normalize_draw_type(s: str) -> str:
    s = (s or "").strip().lower()
    if "mid" in s: return "Midday"
    if "eve" in s: return "Evening"
    return s.title() if s else ""

def est_today():
    est = pytz.timezone("US/Eastern")
    return datetime.now(est).date()

def to_date(datestr: str):
    return datetime.strptime(datestr, "%B %d, %Y").date()

def already_in_sheet(ws, row):
    """Check if (date, draw) already present in the sheet."""
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return False
    df.columns = df.columns.str.strip().str.lower()
    if not {"date", "draw"}.issubset(set(df.columns)):
        return False
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()
    want_date = row["date"]
    want_draw = row["draw"]
    return ((df["date"] == want_date) & (df["draw"] == want_draw)).any()

def upsert_latest(ws, items):
    """
    items: list of parsed dicts (each: date_str, draw, num1, num2, num3, fireball)
    Convert to (date, draw) and append only if not present.
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

# ---------------------------------------------------------------------
# Playwright scraping helpers
# ---------------------------------------------------------------------
def _first_text(el, selectors):
    """Return the first non-empty inner_text found by trying selectors on this element."""
    for sel in selectors:
        try:
            q = el.query_selector(sel)
            if q:
                t = q.inner_text().strip()
                if t:
                    return t
        except Exception:
            pass
    return ""

def _digits(s, n=None):
    d = [c for c in (s or "") if c.isdigit()]
    if n is not None:
        if len(d) < n:
            d = (["0"] * (n - len(d))) + d
        if len(d) > n:
            d = d[:n]
    return d

def debug_dump(page, label="debug"):
    """Save HTML + screenshot to artifacts/ for inspection on failures."""
    try:
        os.makedirs("artifacts", exist_ok=True)
        html_path = f"artifacts/{label}.html"
        img_path  = f"artifacts/{label}.png"
        page.screenshot(path=img_path, full_page=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(page.content())
        print(f"[debug] Saved {html_path} and {img_path}")
    except Exception as e:
        print("[debug] Failed to dump artifacts:", e)

def wait_for_results(page, timeout=60000):
    """Try several card selectors; return the one that appeared."""
    for sel in CARD_SELECTOR_CANDIDATES:
        try:
            page.wait_for_selector(sel, timeout=timeout, state="visible")
            return sel
        except PWTimeout:
            continue
    debug_dump(page, "no-cards")
    raise PWTimeout(f"No result cards visible after {timeout}ms using candidates: {CARD_SELECTOR_CANDIDATES}")

def accept_cookies_if_present(page):
    for sel in [
        "#onetrust-accept-btn-handler",
        "button:has-text('Accept')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
        "button[aria-label*='accept' i]",
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
    # date text (we’ll also regex-scan the whole card text as a fallback)
    date_text = _first_text(card, DATE_CANDIDATES)

    # draw label
    draw_text = normalize_draw_type(_first_text(card, DRAW_LABEL_CANDIDATES))

    # pick3 numbers (3 digits)
    pick_text = _first_text(card, PICK3_NUMS_CANDIDATES)
    pick_digits = _digits(pick_text, n=3)
    pick3 = "".join(pick_digits)

    # fireball (1 digit)
    fb_text = _first_text(card, FIREBALL_CANDIDATES)
    fb_digits = _digits(fb_text, n=1)
    fireball = fb_digits[0] if fb_digits else ""

    # Fallbacks using card’s full text
    full = (card.inner_text() or "").strip()
    if not date_text:
        mdate = DATE_RE.search(full)
        if mdate:
            date_text = mdate.group(0)

    if not draw_text:
        if re.search(r"\bmid(day)?\b", full, re.I):
            draw_text = "Midday"
        elif re.search(r"\beve(ning)?\b", full, re.I):
            draw_text = "Evening"

    return {
        "date_str": date_text,     # <- IMPORTANT: your upsert() expects date_str
        "draw": draw_text,
        "num1": int(pick_digits[0]) if len(pick_digits) >= 1 else 0,
        "num2": int(pick_digits[1]) if len(pick_digits) >= 2 else 0,
        "num3": int(pick_digits[2]) if len(pick_digits) >= 3 else 0,
        "fireball": fireball,
    }

def scrape_latest_cards(max_retries=2):
    """
    Returns a list of items like:
    [{"date_str": "...", "draw": "Midday", "num1": 1, "num2": 2, "num3": 3, "fireball": "4"}, ...]
    Only scrapes the first page (latest results).
    """
    for attempt in range(1, max_retries + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                    timezone_id="America/Chicago",   # IL time
                    viewport={"width": 1280, "height": 2200},
                )

                # Block heavy 3rd-party noise
                context.route("**/*", lambda route: (
                    route.abort()
                    if any(x in route.request.url.lower() for x in [
                        "doubleclick", "googletagmanager", "gtm.js", "analytics",
                        "hotjar", "facebook", "adservice",
                        ".mp4", ".webm", ".gif", ".svg", ".ttf", ".woff", ".woff2"
                    ])
                    else route.continue_()
                ))

                # (Optional) console debug
                context.on("console", lambda msg: print("[console]", msg.type, msg.text))

                page = context.new_page()
                page.set_default_timeout(60000)
                page.goto(PICK3_URL, wait_until="domcontentloaded", timeout=60000)

                accept_cookies_if_present(page)

                used_selector = wait_for_results(page, timeout=60000)

                cards = page.query_selector_all(used_selector)
                if not cards:
                    # Safety net: try all candidates
                    for sel in CARD_SELECTOR_CANDIDATES:
                        cards = page.query_selector_all(sel)
                        if cards:
                            break

                if not cards:
                    debug_dump(page, "cards-empty")
                    raise RuntimeError("No result cards found even after wait + fallbacks.")

                items = [parse_card(c) for c in cards]

                # Filter to rows that have a fireball digit and a 3-digit Pick 3
                filtered = [
                    it for it in items
                    if it.get("fireball", "").isdigit() and isinstance(it.get("num1"), int) and isinstance(it.get("num3"), int)
                ]

                browser.close()

                if not filtered:
                    debug_dump(page, "parsed-empty")
                    raise RuntimeError("Parsed zero usable rows; HTML dumped for inspection.")

                return filtered

        except Exception as e:
            print(f"[scrape] Attempt {attempt} failed: {e}")
            traceback.print_exc()
            if attempt == max_retries:
                raise
            time.sleep(2)  # brief backoff

    return []

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    gc = get_gspread_client()
    ws = open_sheets(gc)

    items = scrape_latest_cards()
    if not items:
        print("No items parsed; nothing to upsert.")
        return

    # Only keep today/yesterday (guards against stale rows)
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

    upsert_latest(ws, keep)

if __name__ == "__main__":
    main()
