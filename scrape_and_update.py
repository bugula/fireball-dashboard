# scrape_and_update.py
import os, json, re, time, traceback
from datetime import datetime, time as dtime
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

# Chicago (= site’s local) clock
CT = pytz.timezone("America/Chicago")
def chicago_now():
    return datetime.now(CT)

def chicago_today():
    return chicago_now().date()

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
    items: list of dicts (date_str, draw, num1, num2, num3, fireball)
    Convert to (date, draw) and append only if not present.
    """
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
            ws.append_row([
                str(row_obj["date"]),
                row_obj["draw"],
                row_obj["num1"],
                row_obj["num2"],
                row_obj["num3"],
                row_obj["fireball"],
            ])
            appended += 1
            print(f"[upsert] Appended: {row_obj['date']} {row_obj['draw']} {row_obj['num1']}{row_obj['num2']}{row_obj['num3']} + FB {row_obj['fireball']}")
        else:
            print(f"[upsert] Exists already: {row_obj['date']} {row_obj['draw']} — skipped")

    print(f"[upsert] Total appended: {appended}")
    return appended

# ---------------------------------------------------------------------
# Assumption fallback (maps 1 new row to the one missing slot)
# ---------------------------------------------------------------------
def expected_missing_slots(ws):
    """
    Look at the sheet and return a list of (date, draw) the app is still missing
    for today (and possibly the obvious next one). Ordered by likelihood.
    """
    df = pd.DataFrame(ws.get_all_records())
    today = chicago_today()
    if df.empty:
        return [(today, "Midday")]  # bootstrap

    df.columns = df.columns.str.strip().str.lower()
    if not {"date","draw"}.issubset(df.columns):
        return [(today, "Midday")]

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()

    def have(d, draw):
        return ((df["date"] == d) & (df["draw"] == draw)).any()

    have_mid_today = have(today, "Midday")
    have_eve_today = have(today, "Evening")

    now_ct = chicago_now().time()
    midday_cut  = dtime(13, 35)  # 1:35pm CT
    evening_cut = dtime(22, 15)  # 10:15pm CT

    missing = []

    # After a cutoff, if we don't have the slot, that's expected next
    if now_ct >= midday_cut and not have_mid_today:
        missing.append((today, "Midday"))

    if now_ct >= evening_cut and not have_eve_today:
        missing.append((today, "Evening"))

    # Between cuts, if Midday exists but Evening doesn't, Evening is expected
    if have_mid_today and not have_eve_today:
        missing.append((today, "Evening"))

    # De-dup while preserving order
    seen = set()
    dedup = []
    for tup in missing:
        if tup not in seen:
            seen.add(tup)
            dedup.append(tup)
    return dedup

def assign_by_assumption(scraped_items, ws):
    """
    If exactly one slot is clearly missing and exactly one scraped item
    has valid digits (even without date/draw labels), assign it to that slot.
    Returns [one_completed_item] or [].
    """
    usable = []
    for it in scraped_items:
        n1, n2, n3 = it.get("num1"), it.get("num2"), it.get("num3")
        fb = str(it.get("fireball", "")).strip()
        if isinstance(n1, int) and isinstance(n2, int) and isinstance(n3, int) and fb.isdigit():
            usable.append(it)

    missing = expected_missing_slots(ws)
    if len(missing) != 1 or len(usable) != 1:
        return []

    target_date, target_draw = missing[0]
    it = usable[0].copy()
    it["date_str"] = it.get("date_str") or target_date.strftime("%B %d, %Y")
    it["draw"] = target_draw
    print(f"[assume] Assigned scraped row to {(target_date, target_draw)}")
    return [it]

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
    """
    Try hard to extract fields from a result card.
    We avoid padding; if we can't find real digits, we return None.
    """
    def safe_text(el):
        try:
            t = el.inner_text().strip()
            if t:
                return t
        except Exception:
            pass
        try:
            t = el.text_content().strip()
            return t or ""
        except Exception:
            return ""

    # Attempt via targeted sub-elements
    date_text = ""
    for sel in DATE_CANDIDATES:
        q = card.query_selector(sel)
        if q:
            date_text = safe_text(q)
            if date_text:
                break

    draw_text = ""
    for sel in DRAW_LABEL_CANDIDATES:
        q = card.query_selector(sel)
        if q:
            draw_text = safe_text(q)
            if draw_text:
                break
    draw_text = normalize_draw_type(draw_text)

    pick_text = ""
    for sel in PICK3_NUMS_CANDIDATES:
        q = card.query_selector(sel)
        if q:
            pick_text = safe_text(q)
            if pick_text:
                break
    pick_digits = [c for c in pick_text if c.isdigit()]

    fb_text = ""
    for sel in FIREBALL_CANDIDATES:
        q = card.query_selector(sel)
        if q:
            fb_text = safe_text(q)
            if fb_text:
                break
    fb_digits = [c for c in fb_text if c.isdigit()]

    # Fallback: scan the entire card text (sometimes markup is odd)
    full = safe_text(card)
    if not date_text:
        mdate = DATE_RE.search(full)
        if mdate:
            date_text = mdate.group(0)

    if not draw_text:
        if re.search(r"\bmid(day)?\b", full, re.I):
            draw_text = "Midday"
        elif re.search(r"\beve(ning)?\b", full, re.I):
            draw_text = "Evening"

    if len(pick_digits) < 3:
        pick_digits = re.findall(r"\b(\d)\b", full)  # spaced digits
    if len(pick_digits) < 3:
        m = re.search(r"(\d)[^\d]{0,3}(\d)[^\d]{0,3}(\d)", full)
        if m:
            pick_digits = [m.group(1), m.group(2), m.group(3)]

    if len(fb_digits) < 1:
        mfb = re.search(r"Fireball[:\s\-]*\s*(\d)", full, re.I)
        if mfb:
            fb_digits = [mfb.group(1)]

    # Validate
    if not (date_text and draw_text in ("Midday", "Evening") and len(pick_digits) >= 3 and len(fb_digits) >= 1):
        return None

    return {
        "date_str": date_text,
        "draw": draw_text,
        "num1": int(pick_digits[0]),
        "num2": int(pick_digits[1]),
        "num3": int(pick_digits[2]),
        "fireball": fb_digits[0],
    }

def scrape_latest_cards(max_retries=2):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[scrape] Attempt {attempt}...")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                context = browser.new_context(
                    user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
                    locale="en-US",
                    timezone_id="America/Chicago",
                    viewport={"width": 1440, "height": 2600},
                )

                page = context.new_page()
                page.set_default_timeout(70000)
                page.goto(PICK3_URL, wait_until="domcontentloaded", timeout=70000)
                accept_cookies_if_present(page)

                # Nudge content to render
                try:
                    page.wait_for_selector("text=Fireball", timeout=30000)
                except Exception:
                    pass
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1000)
                    page.evaluate("window.scrollTo(0, 0)")
                except Exception:
                    pass

                # Prefer scoping to Pick 3 section
                pick3_section = None
                for sel in ["section:has-text('Pick 3')", "section:has-text('Pick3')", "main"]:
                    try:
                        pick3_section = page.query_selector(sel)
                        if pick3_section:
                            break
                    except Exception:
                        continue

                cards = []
                if pick3_section:
                    for sel in CARD_SELECTOR_CANDIDATES:
                        els = pick3_section.query_selector_all(sel)
                        if els:
                            cards = els
                            print(f"[scrape] Using selector in Pick3 scope: {sel} -> {len(cards)}")
                            break
                if not cards:
                    for sel in CARD_SELECTOR_CANDIDATES:
                        els = page.query_selector_all(sel)
                        if els:
                            cards = els
                            print(f"[scrape] Using global selector: {sel} -> {len(cards)}")
                            break

                if not cards:
                    debug_dump(page, "cards-empty")
                    raise RuntimeError("No result cards found even after multiple selectors.")

                items = []
                for i, c in enumerate(cards[:20], 1):
                    it = parse_card(c)
                    if not it:
                        try:
                            snippet = (c.inner_html() or "")[:240].replace("\n", " ")
                            print(f"[scrape] Card#{i} parse failed. Snippet: {snippet}")
                        except Exception:
                            print(f"[scrape] Card#{i} parse failed. (no snippet)")
                    else:
                        items.append(it)
                        print(f"[scrape] Parsed#{i}: {it}")

                browser.close()

                if not items:
                    raise RuntimeError("Parsed zero usable rows (after fallback parsing).")

                return items

        except Exception as e:
            print(f"[scrape] Attempt {attempt} FAILED: {e}")
            traceback.print_exc()
            if attempt == max_retries:
                raise
            time.sleep(2)

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

    # Strict cleaning path: needs date_str + draw label
    cleaned = []
    for it in items:
        ds = (it.get("date_str") or "").strip()
        draw = (it.get("draw") or "").strip().title()
        n1, n2, n3 = it.get("num1"), it.get("num2"), it.get("num3")
        fb = (it.get("fireball") or "").strip()
        if not ds:
            continue
        if draw not in {"Midday", "Evening"}:
            continue
        if not (isinstance(n1, int) and isinstance(n2, int) and isinstance(n3, int)):
            continue
        if not (fb.isdigit() and len(fb) == 1):
            continue
        cleaned.append(it)

    print(f"[filter] Kept {len(cleaned)} rows after completeness checks.")

    # If strict path produced nothing, try assumption fallback
    if not cleaned:
        print("[assume] Trying assumption-based assignment…")
        assumed = assign_by_assumption(items, ws)
        if assumed:
            cleaned = assumed
        else:
            print("[assume] No unambiguous slot to assign — aborting safely.")
            return

    # Only keep today/yesterday (allow up to 2 days) per Chicago time
    today = chicago_today()
    keep = []
    for it in cleaned:
        try:
            d = to_date(it["date_str"])
        except Exception:
            print(f"[filter] Could not parse date_str: {it.get('date_str')}")
            continue
        delta = (today - d).days
        print(f"[filter] Row {it['draw']} {it['date_str']} → {d} (delta={delta})")
        if delta in (0, 1, 2):  # up to 2 days just in case of late posting
            keep.append(it)

    print(f"[filter] Rows kept for upsert: {len(keep)}")
    if not keep:
        print("Parsed items didn’t match today/yesterday; nothing to upsert.")
        return

    appended = upsert_latest(ws, keep)
    print(f"[main] Done. Appended rows: {appended}")

if __name__ == "__main__":
    main()
