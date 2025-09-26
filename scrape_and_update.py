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
BASE_URL   = "https://www.illinoislottery.com"
PICK3_URL  = f"{BASE_URL}/dbg/results/pick3"

# Broad selector candidates (site markup can change)
CARD_SELECTOR_CANDIDATES = [
    "a.dbg-results__result",
    "li[data-result] a[href*='/dbg/results/pick3/draw/']",
    "section:has-text('Pick 3') a[href*='/dbg/results/pick3/draw/']",
    "a[href*='/dbg/results/pick3/draw/']",
]

# On the DETAIL page we try these:
DETAIL_DATE_SEL   = [
    "[data-testid='draw-date']",
    ".dbg-draw-header__date",
    "time[datetime]", "time"
]
DETAIL_DRAW_SEL   = [
    "[data-testid='draw-name']",
    ".dbg-draw-header__draw",
    "h1,h2,h3:has-text('Midday')",
    "h1,h2,h3:has-text('Evening')",
    "*:text('Midday')",
    "*:text('Evening')",
]
DETAIL_P3_SEL     = [
    "[data-testid='result-number']",
    ".dbg-winning-number",
    ".winning-number",
    ".dbg-winning-numbers, .winning-numbers"
]
DETAIL_FIREBALL_SEL = [
    "[data-testid='fireball']",
    ".dbg-fireball, .fireball",
    "*:has-text('Fireball')"
]

# ---------------------------------------------------------------------
# Google Sheets auth
# ---------------------------------------------------------------------
def get_gspread_client():
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
    sheet_id = os.environ.get("FIREBALL_DATA_SHEET_ID")
    if not sheet_id:
        raise RuntimeError("Missing FIREBALL_DATA_SHEET_ID secret.")
    try:
        ss = gc.open_by_key(sheet_id)
    except gspread.SpreadsheetNotFound:
        raise RuntimeError(
            "Spreadsheet not found. Make sure the GitHub Action's service account "
            "email has at least Editor access to this spreadsheet."
        )
    ws = ss.sheet1
    print(f"[sheets] Opened spreadsheet: {ss.title} (sheet1 title: {ws.title})")
    return ws


# ---------------------------------------------------------------------
# Helpers: parsing & dates
# ---------------------------------------------------------------------
MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
DATE_RE = re.compile(rf"{MONTHS}\s+\d{{1,2}},\s+\d{{4}}", re.I)

CT = pytz.timezone("America/Chicago")
def chicago_now():
    return datetime.now(CT)
def chicago_today():
    return chicago_now().date()

def normalize_draw_type(s: str) -> str:
    s = (s or "").strip().lower()
    if "mid" in s: return "Midday"
    if "eve" in s: return "Evening"
    return s.title() if s else ""

def to_date(datestr: str):
    return datetime.strptime(datestr, "%B %d, %Y").date()

def already_in_sheet(ws, row):
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return False
    df.columns = df.columns.str.strip().str.lower()
    if not {"date","draw"}.issubset(df.columns):
        return False
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()
    return ((df["date"] == row["date"]) & (df["draw"] == row["draw"])).any()

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

# ---------- NEW: last-seen helpers to keep only strictly-new rows ----------
def draw_rank(draw):
    return {"Midday": 0, "Evening": 1}.get(str(draw).strip().title(), -1)

def get_last_seen(ws):
    """
    Return (last_date, last_draw) based on the sheet:
    - last_date is the max date present
    - last_draw is the latest within that date (Evening > Midday)
    """
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return (None, None)
    df.columns = df.columns.str.strip().str.lower()
    if not {"date","draw"}.issubset(df.columns):
        return (None, None)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()
    df = df.dropna(subset=["date"])
    if df.empty:
        return (None, None)
    last_date = df["date"].max()
    same_day = df[df["date"] == last_date].copy()
    if same_day.empty:
        return (last_date, None)
    same_day["rank"] = same_day["draw"].map(draw_rank)
    same_day = same_day.sort_values("rank")  # Midday(0) then Evening(1)
    last_draw = same_day.iloc[-1]["draw"]    # take the highest rank
    return (last_date, last_draw)

def filter_newer_than_sheet(items, ws):
    """
    Keep only items strictly newer than the sheet's last (date, draw).
    Ordering: date asc, within date Midday(0) < Evening(1).
    """
    last_date, last_draw = get_last_seen(ws)
    print(f"[last_seen] Sheet last_date={last_date}, last_draw={last_draw}")

    if last_date is None:
        print("[newer] Sheet empty; keeping all items.")
        return items

    out = []
    last_rank = draw_rank(last_draw)
    for it in items:
        try:
            d = to_date(it["date_str"])
        except Exception:
            print(f"[newer] Bad date_str in item: {it}")
            continue
        r = draw_rank(it.get("draw"))
        if d > last_date:
            out.append(it)
        elif d == last_date and r > last_rank:
            out.append(it)
        else:
            print(f"[newer] Skipping not-new: {it.get('draw')} {it.get('date_str')}")
    print(f"[newer] Candidates after last_seen filter: {len(out)}")
    return out

# ---------- Assumption fallback (1 missing slot) ----------
def expected_missing_slots(ws):
    df = pd.DataFrame(ws.get_all_records())
    today = chicago_today()
    if df.empty:
        return [(today, "Midday")]  # bootstrap
    df.columns = df.columns.str.strip().str.lower()
    if not {"date","draw"}.issubset(df.columns):
        return [(today, "Midday")]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["draw"] = df["draw"].astype(str).str.strip().str.title()
    def have(d, draw): return ((df["date"] == d) & (df["draw"] == draw)).any()

    have_mid_today = have(today, "Midday")
    have_eve_today = have(today, "Evening")
    now_ct = chicago_now().time()
    midday_cut  = dtime(13, 35)  # 1:35pm CT
    evening_cut = dtime(22, 15)  # 10:15pm CT

    missing = []
    if now_ct >= midday_cut and not have_mid_today:
        missing.append((today, "Midday"))
    if now_ct >= evening_cut and not have_eve_today:
        missing.append((today, "Evening"))
    if have_mid_today and not have_eve_today:
        missing.append((today, "Evening"))

    seen, dedup = set(), []
    for m in missing:
        if m not in seen:
            seen.add(m); dedup.append(m)
    return dedup

def assign_by_assumption(scraped_items, ws):
    usable = []
    for it in scraped_items:
        n1, n2, n3 = it.get("num1"), it.get("num2"), it.get("num3")
        fb = str(it.get("fireball","")).strip()
        if isinstance(n1,int) and isinstance(n2,int) and isinstance(n3,int) and fb.isdigit():
            usable.append(it)
    missing = expected_missing_slots(ws)
    if len(missing) != 1 or len(usable) != 1:
        return []
    target_date, target_draw = missing[0]
    it = usable[0].copy()
    it["date_str"] = it.get("date_str") or target_date.strftime("%B %d, %Y")
    it["draw"]     = it.get("draw")     or target_draw
    print(f"[assume] Assigned scraped row to {(target_date, target_draw)}")
    return [it]

# ---------------------------------------------------------------------
# Playwright scraping helpers
# ---------------------------------------------------------------------
def safe_text(el):
    if not el: return ""
    try:
        t = el.inner_text().strip()
        if t: return t
    except Exception:
        pass
    try:
        t = el.text_content().strip()
        return t or ""
    except Exception:
        return ""

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
            btn = page.query_selector(sel)
            if btn:
                btn.click()
                page.wait_for_timeout(300)
                return
        except Exception:
            pass

def collect_draw_links(page):
    """Return absolute hrefs for recent Pick 3 draws on the LIST page."""
    for sel in CARD_SELECTOR_CANDIDATES:
        links = page.query_selector_all(sel)
        if links:
            hrefs = []
            for a in links[:12]:  # first 12 rows is plenty
                try:
                    href = a.get_attribute("href") or ""
                    if "/dbg/results/pick3/draw/" in href:
                        if href.startswith("http"):
                            hrefs.append(href)
                        else:
                            hrefs.append(BASE_URL + href)
                except Exception:
                    continue
            if hrefs:
                print(f"[links] Using selector {sel}: {len(hrefs)} links")
                return hrefs
    return []

def parse_detail_page(page):
    """
    Parse from a DRAW DETAIL page.
    Returns dict or None.
    """
    def _s(el):
        return safe_text(el)

    try:
        body_text = page.locator("body").inner_text(timeout=0)
    except Exception:
        body_text = ""
    html = page.content()

    # --- Date ---
    date_text = ""
    for sel in DETAIL_DATE_SEL:
        date_text = _s(page.query_selector(sel))
        if date_text:
            break
    if not date_text:
        mdate = DATE_RE.search(html) or DATE_RE.search(body_text)
        if mdate:
            date_text = mdate.group(0)

    # --- Draw type ---
    draw_text = ""
    for sel in DETAIL_DRAW_SEL:
        t = _s(page.query_selector(sel))
        if t:
            draw_text = normalize_draw_type(t)
            if draw_text in ("Midday", "Evening"):
                break
    if not draw_text:
        if re.search(r"\bmid(day)?\b", html, re.I) or re.search(r"\bmid(day)?\b", body_text, re.I):
            draw_text = "Midday"
        elif re.search(r"\beve(ning)?\b", html, re.I) or re.search(r"\beve(ning)?\b", body_text, re.I):
            draw_text = "Evening"

    # --- Pick 3 digits ---
    nums = []
    for sel in DETAIL_P3_SEL:
        block = _s(page.query_selector(sel))
        if block:
            nums.extend([c for c in block if c.isdigit()])
    if len(nums) < 3:
        m = re.search(r"Winning\s*Numbers?.*?(\d)[^\d]{0,3}(\d)[^\d]{0,3}(\d)", html, re.I | re.S)
        if not m:
            m = re.search(r"Winning\s*Numbers?.*?(\d)[^\d]{0,3}(\d)[^\d]{0,3}(\d)", body_text, re.I | re.S)
        if m:
            nums = [m.group(1), m.group(2), m.group(3)]
    if len(nums) < 3 and body_text:
        m = re.search(r"(\d)[^\d]{0,3}(\d)[^\d]{0,3}(\d)", body_text)
        if m:
            nums = [m.group(1), m.group(2), m.group(3)]
    nums = nums[:3]

    # --- Fireball ---
    fb = ""
    for sel in DETAIL_FIREBALL_SEL:
        block = _s(page.query_selector(sel))
        if block:
            fb_digits = re.findall(r"\d", block)
            if fb_digits:
                fb = fb_digits[0]
                break
    if not fb:
        mfb = re.search(r"Fireball[:\s\-]*\s*(\d)", html, re.I) or re.search(r"Fireball[:\s\-]*\s*(\d)", body_text, re.I)
        if mfb:
            fb = mfb.group(1)

    if date_text and draw_text in ("Midday", "Evening") and len(nums) == 3 and fb.isdigit():
        return {
            "date_str": date_text,
            "draw": draw_text,
            "num1": int(nums[0]),
            "num2": int(nums[1]),
            "num3": int(nums[2]),
            "fireball": fb,
        }
    return None


def scrape_latest_cards(max_retries=2):
    """
    Strategy:
    1) Open list page, collect the draw links.
    2) Visit top N detail pages and parse text (date/draw/numbers/fireball).
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[scrape] Attempt {attempt}…")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage"])
                context = browser.new_context(
                    user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/124.0.0.0 Safari/537.36"),
                    locale="en-US",
                    timezone_id="America/Chicago",
                    viewport={"width": 1440, "height": 2600},
                )

                page = context.new_page()
                page.set_default_timeout(70000)
                page.goto(PICK3_URL, wait_until="domcontentloaded", timeout=70000)
                accept_cookies_if_present(page)

                # light nudge
                try:
                    page.wait_for_selector("text=Pick 3", timeout=15000)
                except Exception:
                    pass

                links = collect_draw_links(page)
                if not links:
                    print("[links] No draw links found on list page.")
                    return []

                print(f"[links] Visiting {min(8,len(links))} detail pages…")
                items = []
                for i, href in enumerate(links[:8], 1):
                    try:
                        page.goto(href, wait_until="domcontentloaded", timeout=70000)
                        accept_cookies_if_present(page)
                        try: page.wait_for_selector("text=Fireball", timeout=5000)
                        except Exception: pass
                        it = parse_detail_page(page)
                        if it:
                            items.append(it)
                            print(f"[detail] Parsed {i}: {it}")
                        else:
                            print(f"[detail] Could not parse page {i}: {href}")
                    except Exception as e:
                        print(f"[detail] Error on {href}: {e}")

                browser.close()

                if not items:
                    raise RuntimeError("Parsed zero usable rows from detail pages.")

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
    # 1) Sheets
    gc = get_gspread_client()
    ws = open_sheets(gc)

    # --- TEMP healthcheck write (remove after verifying) ---
    try:
        ts = chicago_now().strftime("%Y-%m-%d %H:%M:%S %Z")
        ws.update("H1", f"GitHub Action connected @ {ts}")
        print("[healthcheck] Wrote connection stamp to H1.")
    except Exception as e:
        print("[healthcheck] Failed to write to sheet:", e)
        raise
    # --- end TEMP ---
    
    # 2) Scrape
    items = scrape_latest_cards()
    if not items:
        print("No items parsed; nothing to upsert.")
        return
    print(f"[main] Parsed {len(items)} items:")
    for it in items:
        print("       ", it)

    # 3) STRICT completeness check
    cleaned = []
    for it in items:
        ds = (it.get("date_str") or "").strip()
        draw = (it.get("draw") or "").strip().title()
        n1, n2, n3 = it.get("num1"), it.get("num2"), it.get("num3")
        fb = (it.get("fireball") or "").strip()
        if not ds: continue
        if draw not in {"Midday","Evening"}: continue
        if not (isinstance(n1,int) and isinstance(n2,int) and isinstance(n3,int)): continue
        if not (fb.isdigit() and len(fb) == 1): continue
        cleaned.append(it)
    print(f"[filter] Kept {len(cleaned)} rows after completeness checks.")

    # 4) If nothing strict, try assumption fallback (only when unambiguous)
    if not cleaned:
        print("[assume] Trying assumption-based assignment…")
        assumed = assign_by_assumption(items, ws)
        if assumed:
            cleaned = assumed
            print(f"[assume] Using assumed row: {assumed[0]}")
        else:
            print("[assume] No unambiguous slot to assign — aborting safely.")
            return

    # 5) Keep only rows strictly newer than what’s already in the sheet
    keep = filter_newer_than_sheet(cleaned, ws)
    if not keep:
        print("[main] Nothing newer than sheet; no append.")
        return

    # 6) Append (upsert re-checks duplicates just in case)
    appended = upsert_latest(ws, keep)
    print(f"[main] Done. Appended rows: {appended}")

if __name__ == "__main__":
    main()
