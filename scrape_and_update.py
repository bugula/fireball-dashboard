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
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        json.loads(creds_json),
        scope
    )
    return gspread.authorize(creds)
