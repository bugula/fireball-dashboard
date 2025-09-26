# scrape_and_update.py
import os, json, re, time, traceback
from datetime import datetime, time as dtime
import pytz

from playwright.sync_api import sync_playwright
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# =========================
# Debug artifacts (unchanged)
# =========================
import json as _json
from pathlib import Path

ART_DIR = Path("artifacts")
def _ensure_art_dir():
    ART_DIR.mkdir(parents=True, exist_ok=True)

def save_text(name: str, text: str):
    _ensure_art_dir()
    p = ART_DIR / name
    p.write_text(text or "", encoding="utf-8")
    print(f"[artifact] wrote {p}")

def save_json(name: str, obj):
    _ensure_art_dir()
    p = ART_DIR / name
    p.write_text(_json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[artifact] wrote {p}")

def save_screenshot(page, name: str):
    try:
        _ensure_art_dir()
        p = ART_DIR / name
        page.screenshot(path=str(p), full_page=True)
        print(f"[artifact] wrote {p}")
    except Exception as e:
        print(f"[artifact] screenshot failed: {e}")

# =========================
# Config
# =========================
BASE_URL   = "https://www.illinoislottery.com"
PICK3_URL  = f"{BASE_URL}/dbg/results/pick3"

CARD_SELECTOR_CANDIDATES = [
    "a.dbg-results__result",
    "li[data-result] a[href*='/dbg/results/pick3/draw/']",
    "section:has-text('Pick 3') a[href*='/dbg/results/pick3/draw/']",
    "a[href*='/dbg/results/pick3/draw/']",
]

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

# Date helpers
MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
DATE_RE = re.compile(rf"{MONTHS}\s+\d{{1,2}},\s+\d{{4}}", re.I)

CT = pytz.timezone("America/Chicago")
def chicago_now():
    return datetime.now(CT)
def chicago_today():
    return chicago_now().date()

# =========================
# Google Sheets
# =========================
def get_gspread_client():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT")
    if not creds_json:
        b64 = os.environ.get("GCP_SERVICE_ACCOUNT_B64")
        if b64:
            import base64
            creds_json = base64.b64decode(b64).decode("utf-8")
    if not creds_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT (or GCP_SERVICE_ACCOUNT_B64) secret.")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(creds_json), scope)
    return gspread.authorize(creds)

def open_sheets(gc):
    sheet_id = os.environ.get("FIREBALL_DATA_SHEET_ID", "").strip()
    if sheet_id:
        ss = gc.open_by_key(sheet_id)
    else:
        ss = gc.open("fireball_data")
    ws = ss.sheet1
    print(f"[sheets] Opened spreadsheet: {ss.title} (sheet1: {ws.title})")
    return ws

# =========================
# Helpers
# =========================
def normalize_draw_type(s: str) -> str:
    s = (s or "").strip().lower()
    if "mid" in s: return "Midday"
    if "eve" in s: return "Evening"
    return s.title() if s else ""

def _parse_date_str(raw: str):
    """
    Accept inputs like:
      'Thursday,\nSep 25, 2025\nevening'
      'Sep 25, 2025'
      'September 25, 2025'
    Return a date() or raise.
    """
    if not raw:
        raise ValueError("empty date_str")
    s = " ".join(raw.split())  # collapse newlines/spaces
    # strip weekday and trailing words like 'evening/midday'
    # find the 'Mon DD, YYYY' or 'Month DD, YYYY' core
    m = DATE_RE.search(s)
    if not m:
        raise ValueError(f"no month-date-year found in: {s!r}")
    core = m.group(0)  # e.g. 'Sep 25, 2025' or 'September 25, 2025'
    # Try several formats
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(core, fmt).date()
        except Exception:
            pass
    raise ValueError(f"unparsed date core: {core!r}")

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
        print(f"[dup] Row already present: {row['date']} {row['draw']}")
    return exists

def upsert_latest(ws, items):
    appended = 0
    for it in items:
        try:
            d = _parse_date_str(it["date_str"])
        except Exception as e:
            print(f"[upsert] Bad date: {it.get('date_str')} ({e}), skipping")
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
            print(f"[append] WILL APPEND -> {row_obj['date']} {row_obj['draw']} "
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
            print(f"[append] Skipped (exists) -> {row_obj['date']} {row_obj['draw']}")
    print(f"[upsert] Total appended: {appended}")
    return appended

# ---------- Assumption fallback (only if exactly one slot is missing) ----------
def expected_missing_slots(ws):
    df = pd.DataFrame(ws.get_all_records())
    today = chicago_today()
    if df.empty:
        return [(today, "Midday")]
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
    it["date_str"] = it.get("date_str") or target_date.strftime("%b %d, %Y")
    it["draw"]     = it.get("draw")     or target_draw
    print(f"[assume] Assigned scraped row to {(target_date, target_draw)}")
    return [it]

# =========================
# Playwright scraping
# =========================
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

def collect_draw_links(page):
    for sel in CARD_SELECTOR_CANDIDATES:
        links = page.query_selector_all(sel)
        if links:
            hrefs = []
            for a in links[:12]:
                try:
                    href = a.get_attribute("href") or ""
                    if "/dbg/results/pick3/draw/" in href:
                        hrefs.append(href if href.startswith("http") else (BASE_URL + href))
                except Exception:
                    continue
            if hrefs:
                print(f"[links] Using selector {sel}: {len(hrefs)} links")
                return hrefs
    return []

def _extract_numbers_from_accessibility(html: str, text: str):
    """
    Prefer accessibility strings like:
      'Winning number 1 is 7'
      'Winning number 2 is 5'
      'Winning number 3 is 8'
    Returns list[str] of length 3 or [].
    """
    blob = (html or "") + "\n" + (text or "")
    # Normalize spacing
    blob = " ".join(blob.split())
    p = re.compile(r"Winning\s+number\s*1\s*is\s*(\d).*?Winning\s+number\s*2\s*is\s*(\d).*?Winning\s+number\s*3\s*is\s*(\d)", re.I)
    m = p.search(blob)
    if m:
        return [m.group(1), m.group(2), m.group(3)]
    # Alternative phrasing: 'Winning number one is 7' is rare, but keep numeric only.
    return []

def _extract_numbers_fallback(html: str, text: str):
    """
    Tight fallback: look for 'Winning Numbers:' followed by three single digits with small separators.
    """
    blob = (html or "") + "\n" + (text or "")
    blob = " ".join(blob.split())
    m = re.search(r"Winning\s*Numbers?.*?(\d)[^\d]{0,3}(\d)[^\d]{0,3}(\d)", blob, re.I)
    if m:
        return [m.group(1), m.group(2), m.group(3)]
    # As a last resort: any three single digits close together (but we’ll sanity-check later)
    m2 = re.search(r"(\d)[^\d]{0,2}(\d)[^\d]{0,2}(\d)", blob)
    if m2:
        return [m2.group(1), m2.group(2), m2.group(3)]
    return []

def _extract_fireball(html: str, text: str):
    blob = (html or "") + "\n" + (text or "")
    blob = " ".join(blob.split())
    m = re.search(r"Fireball[:\s\-]*\s*(\d)", blob, re.I)
    return m.group(1) if m else ""

def parse_detail_page(page):
    # robust blobs
    try:
        body_text = page.locator("body").inner_text(timeout=0)
    except Exception:
        body_text = ""
    html = page.content()

    # Date
    date_text = ""
    for sel in DETAIL_DATE_SEL:
        date_text = safe_text(page.query_selector(sel))
        if date_text:
            break
    if not date_text:
        mdate = DATE_RE.search(html) or DATE_RE.search(body_text)
        if mdate:
            date_text = mdate.group(0)

    # Draw type
    draw_text = ""
    for sel in DETAIL_DRAW_SEL:
        t = safe_text(page.query_selector(sel))
        if t:
            t = normalize_draw_type(t)
            if t in ("Midday","Evening"):
                draw_text = t
                break
    if not draw_text:
        if re.search(r"\bmid(day)?\b", html, re.I) or re.search(r"\bmid(day)?\b", body_text, re.I):
            draw_text = "Midday"
        elif re.search(r"\beve(ning)?\b", html, re.I) or re.search(r"\beve(ning)?\b", body_text, re.I):
            draw_text = "Evening"

    # Numbers – STRONG path first: accessibility text
    nums = _extract_numbers_from_accessibility(html, body_text)

    # If that yielded nothing, try targeted number containers (but DO NOT pad)
    if not nums:
        blocks = []
        for sel in DETAIL_P3_SEL:
            b = safe_text(page.query_selector(sel))
            if b:
                blocks.append(b)
        joined = " ".join(blocks)
        cand = re.findall(r"\b(\d)\b", joined)
        if len(cand) >= 3:
            nums = cand[:3]

    # If still nothing, use tight fallback
    if not nums:
        nums = _extract_numbers_fallback(html, body_text)

    # Fireball
    fb = ""
    for sel in DETAIL_FIREBALL_SEL:
        block = safe_text(page.query_selector(sel))
        if block:
            fb_digits = re.findall(r"\d", block)
            if fb_digits:
                fb = fb_digits[0]
                break
    if not fb:
        fb = _extract_fireball(html, body_text)

    # Guard against classic placeholder "1 2 3" – re-try only with accessibility pattern
    if nums == ["1","2","3"]:
        print("[parse] WARNING: got 1,2,3 — rechecking with accessibility-only path.")
        nums2 = _extract_numbers_from_accessibility(html, body_text)
        if nums2 and nums2 != ["1","2","3"]:
            nums = nums2

    if not (date_text and draw_text in ("Midday","Evening") and len(nums) == 3 and fb.isdigit()):
        print(f"[parse] Incomplete parse -> date={date_text!r}, draw={draw_text!r}, nums={nums}, fb={fb!r}")
        return None

    out = {
        "date_str": date_text,
        "draw": draw_text,
        "num1": int(nums[0]),
        "num2": int(nums[1]),
        "num3": int(nums[2]),
        "fireball": fb,
    }
    print(f"[parse] OK -> {out}")
    return out

def scrape_latest_cards(max_retries=2):
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

                # dump list page
                try:
                    save_text("list.html", page.content())
                    save_screenshot(page, "list.png")
                except Exception:
                    pass

                try: page.wait_for_selector("text=Pick 3", timeout=15000)
                except Exception: pass

                links = collect_draw_links(page)
                if not links:
                    print("[links] No draw links found.")
                    return []

                print(f"[links] Visiting {min(8,len(links))} detail pages…")
                items = []
                for i, href in enumerate(links[:8], 1):
                    try:
                        page.goto(href, wait_until="domcontentloaded", timeout=70000)
                        try: page.wait_for_selector("text=Fireball", timeout=5000)
                        except Exception: pass
                        save_text(f"detail_{i}.html", page.content())
                        save_screenshot(page, f"detail_{i}.png")

                        it = parse_detail_page(page)
                        if it:
                            items.append(it)
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

# =========================
# Main
# =========================
def main():
    gc = get_gspread_client()
    ws = open_sheets(gc)

    items = scrape_latest_cards()

    # DEBUG mode: skip writing
    if os.getenv("DEBUG_SCRAPE_ONLY", "0") == "1":
        print(f"[debug] DEBUG_SCRAPE_ONLY=1 -> not writing to Sheets.")
        print(f"[debug] items scraped: {len(items)}")
        save_json("items_raw.json", items)
        return

    if not items:
        print("No items parsed; nothing to upsert.")
        return

    # Strict filter (all fields present)
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

    if not cleaned:
        print("[assume] Trying assumption-based assignment…")
        assumed = assign_by_assumption(items, ws)
        if assumed:
            cleaned = assumed
        else:
            print("[assume] No unambiguous slot to assign — aborting safely.")
            return

    # Keep only recent (today/yesterday, allow 2 days buffer)
    today = chicago_today()
    keep = []
    for it in cleaned:
        try:
            d = _parse_date_str(it["date_str"])
        except Exception as e:
            print(f"[filter] Could not parse date_str: {it.get('date_str')} ({e})")
            continue
        delta = (today - d).days
        print(f"[filter] Row {it['draw']} {it['date_str']} → {d} (delta={delta})")
        if delta in (0,1,2):
            keep.append(it)

    print(f"[filter] Rows kept for upsert: {len(keep)}")
    if not keep:
        print("Parsed items didn’t match today/yesterday; nothing to upsert.")
        return

    appended = upsert_latest(ws, keep)
    if appended == 0:
        print("[result] 0 rows appended. Likely they already existed, or the service account lacks write access.")
        print("         If your Streamlit app writes fine but this doesn’t, SHARE the sheet with the "
              "service account’s client_email from the Action’s JSON.")
    else:
        print(f"[result] Appended {appended} new row(s).")

if __name__ == "__main__":
    main()
