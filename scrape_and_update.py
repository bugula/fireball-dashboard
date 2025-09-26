# scrape_and_update.py
import os, json, re, time, traceback
from datetime import datetime, time as dtime
import pytz

from playwright.sync_api import sync_playwright
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# =========================
# Debug artifacts
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

# List page → draw links
CARD_SELECTOR_CANDIDATES = [
    "a.dbg-results__result",
    "li[data-result] a[href*='/dbg/results/pick3/draw/']",
    "section:has-text('Pick 3') a[href*='/dbg/results/pick3/draw/']",
    "a[href*='/dbg/results/pick3/draw/']",
]

# Detail page selectors
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

# Container(s) that actually hold the 3 winning digits
WINNUMS_CONTAINER_SEL = [
    "[data-testid='winning-numbers']",
    ".dbg-winning-numbers",
    ".winning-numbers",
    "section:has-text('Winning Numbers')",
    "div:has-text('Winning Numbers')",
]

# Inside the container, places to look for digits
WINNUMS_DIGIT_SEL = [
    "[aria-label*='Winning number' i]",
    "[alt]",
    "[data-testid='result-number']",
    ".dbg-winning-number",
    ".winning-number",
    "span, div, li, img",
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
    m = DATE_RE.search(s)
    if not m:
        raise ValueError(f"no month-date-year found in: {s!r}")
    core = m.group(0)  # 'Sep 25, 2025' or 'September 25, 2025'
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

def _numbers_from_container(container):
    """Extract 3 digits strictly from the winning-numbers container."""
    if not container:
        return []
    # 1) Accessibility first: 'Winning number N is X'
    blob = " ".join((container.inner_text() or "").split())
    m = re.search(r"Winning\s+number\s*1\s*is\s*(\d).*?Winning\s+number\s*2\s*is\s*(\d).*?Winning\s+number\s*3\s*is\s*(\d)", blob, re.I)
    if m:
        return [m.group(1), m.group(2), m.group(3)]

    # 2) Element attributes inside the container
    digits = []
    for sel in WINNUMS_DIGIT_SEL:
        for node in container.query_selector_all(sel):
            # prefer aria-label / alt
            txt = (node.get_attribute("aria-label") or
                   node.get_attribute("alt") or
                   safe_text(node) or "")
            m1 = re.search(r"\b(\d)\b", txt)
            if m1:
                digits.append(m1.group(1))
            if len(digits) >= 3:
                return digits[:3]

    # 3) Tight fallback INSIDE the container only
    html = container.inner_html() or ""
    html = " ".join(html.split())
    m2 = re.search(r"(\d)[^\d]{0,2}(\d)[^\d]{0,2}(\d)", html)
    if m2:
        return [m2.group(1), m2.group(2), m2.group(3)]

    return []

def _fireball_from_page(page, container=None):
    """Find Fireball digit, prefer container vicinity first."""
    # try near-container
    if container:
        txt = safe_text(container)
        m = re.search(r"Fireball[:\s\-]*\s*(\d)", txt, re.I)
        if m: return m.group(1)

    # fall back to page
    for sel in DETAIL_FIREBALL_SEL:
        block = safe_text(page.query_selector(sel))
        if block:
            fb_digits = re.findall(r"\d", block)
            if fb_digits:
                return fb_digits[0]

    # final regex on whole page
    body = safe_text(page.locator("body"))
    m = re.search(r"Fireball[:\s\-]*\s*(\d)", body, re.I)
    return m.group(1) if m else ""

def parse_detail_page(page):
    """Parse a detail page into a dict or None using text windows near headings."""
    # ---- grab big text blobs once
    try:
        body_text = page.locator("body").inner_text(timeout=0)
    except Exception:
        body_text = ""
    html = page.content()

    # ---- date (use your robust helper downstream)
    # pull a broad date candidate string (we'll parse with _parse_date_str)
    date_text = ""
    # try common date spots
    for sel in ["[data-testid='draw-date']", ".dbg-draw-header__date", "time[datetime]", "time"]:
        t = safe_text(page.query_selector(sel))
        if t:
            date_text = t
            break
    if not date_text:
        mdate = DATE_RE.search(html) or DATE_RE.search(body_text)
        if mdate:
            date_text = mdate.group(0)
        else:
            # we will still try to parse later; keep empty if none
            date_text = ""

    # ---- draw type
    draw_text = ""
    for sel in ["[data-testid='draw-name']", ".dbg-draw-header__draw"]:
        t = safe_text(page.query_selector(sel))
        if t:
            t = normalize_draw_type(t)
            if t in ("Midday", "Evening"):
                draw_text = t
                break
    if not draw_text:
        bt = body_text.lower()
        if "midday" in bt:
            draw_text = "Midday"
        elif "evening" in bt:
            draw_text = "Evening"

    # ---- narrow to window after "Winning Numbers"
    bt_compact = " ".join(body_text.split())
    idx = re.search(r"winning\s+numbers", bt_compact, re.I)
    window = ""
    if idx:
        start = idx.start()
        window = bt_compact[start:start + 1200]  # small, local window
    else:
        # fallback: look for "Winning number 1 is"
        idx2 = re.search(r"winning\s+number\s*1", bt_compact, re.I)
        if idx2:
            start = idx2.start()
            window = bt_compact[start:start + 1200]

    # ---- extract 3 digits from the window ONLY
    nums = []
    if window:
        # Accessibility style first: "Winning number 1 is X"
        m = re.search(
            r"winning\s*number\s*1\s*is\s*(\d).*?winning\s*number\s*2\s*is\s*(\d).*?winning\s*number\s*3\s*is\s*(\d)",
            window, re.I
        )
        if m:
            nums = [m.group(1), m.group(2), m.group(3)]
        if len(nums) < 3:
            # tight 3 single-digits near each other (avoid dates by staying inside window)
            m2 = re.search(r"(\d)[^\d]{0,2}(\d)[^\d]{0,2}(\d)", window)
            if m2:
                nums = [m2.group(1), m2.group(2), m2.group(3)]

    # ---- fireball: small window around the word Fireball
    fb = ""
    fb_idx = re.search(r"fireball", bt_compact, re.I)
    if fb_idx:
        start = max(0, fb_idx.start() - 60)
        end   = fb_idx.start() + 120
        fb_window = bt_compact[start:end]
        mfb = re.search(r"fireball[:\s\-]*\s*(\d)", fb_window, re.I)
        if mfb:
            fb = mfb.group(1)
    if not fb:
        # last resort: look for a single digit right after Fireball anywhere
        mfb2 = re.search(r"fireball[:\s\-]*\s*(\d)", bt_compact, re.I)
        if mfb2:
            fb = mfb2.group(1)

    # ---- validate / finalize
    if not date_text or draw_text not in ("Midday", "Evening") or len(nums) != 3 or not fb.isdigit():
        print(f"[parse-winwin] Incomplete -> date={date_text!r}, draw={draw_text!r}, nums={nums}, fb={fb!r}")
        return None

    # guard against the infamous 1,2,3 placeholder unless explicitly stated
    if nums == ["1","2","3"]:
        acc_ok = re.search(
            r"winning\s*number\s*1\s*is\s*1.*?winning\s*number\s*2\s*is\s*2.*?winning\s*number\s*3\s*is\s*3",
            window, re.I
        )
        if not acc_ok:
            print("[parse-winwin] Rejected 1,2,3 (likely placeholder).")
            return None

    out = {
        "date_str": date_text,
        "draw": draw_text,
        "num1": int(nums[0]),
        "num2": int(nums[1]),
        "num3": int(nums[2]),
        "fireball": fb,
    }
    print(f"[parse-winwin] OK -> {out}")
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

    if os.getenv("DEBUG_SCRAPE_ONLY", "0") == "1":
        print(f"[debug] DEBUG_SCRAPE_ONLY=1 -> not writing to Sheets.")
        print(f"[debug] items scraped: {len(items)}")
        save_json("items_raw.json", items)
        return

    if not items:
        print("No items parsed; nothing to upsert.")
        return

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
    else:
        print(f"[result] Appended {appended} new row(s).")

if __name__ == "__main__":
    main()
