
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the scraper metadata CSV, find newly added docs since the last run,
compose 3 tweet variants per doc (NL), and post to X (Twitter) using Tweepy.
Falls back to dry-run (prints) if no credentials provided or DRY_RUN=1.
"""
import os, json, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd

# ----- Config -----
YEAR = os.getenv("RIS_YEAR", str(datetime.now().year))
META_CSV = Path(f"./data/denhaag/{YEAR}/documents_metadata.csv")
STATE_DIR = Path("./state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "seen_ids.json"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

def load_seen():
    if STATE_FILE.exists():
        try:
            return set(json.loads(STATE_FILE.read_text()))
        except Exception:
            return set()
    return set()

def save_seen(seen):
    STATE_FILE.write_text(json.dumps(sorted(seen)))

def short_date(dt):
    if pd.isna(dt):
        return ""
    try:
        return pd.to_datetime(dt).strftime("%-d %b %Y")
    except Exception:
        return str(dt)[:10]

def mk_variants(row):
    mt = (row.get("meeting_title") or "").strip()
    md = short_date(row.get("meeting_date"))
    dt = (row.get("doc_type") or "document").lower()
    dl = (row.get("doc_label") or "").strip()
    url = (row.get("doc_url") or "").strip()
    # Verkorte 'titel' voor in de tweet
    title = dl if dl else dt.capitalize()
    # Hashtags basaal; kan je naar wens aanpassen
    tags_map = {
        "notulen": ["#DenHaag", "#Gemeenteraad", "#Notulen"],
        "besluitenlijst": ["#DenHaag", "#Gemeenteraad", "#Besluiten"],
        "motie": ["#DenHaag", "#Motie"],
        "amendement": ["#DenHaag", "#Amendement"],
        "raadsvoorstel": ["#DenHaag", "#Raad", "#Voorstel"],
        "agenda": ["#DenHaag", "#Agenda"],
        "bijlage": ["#DenHaag", "#Bijlage"],
        "overig": ["#DenHaag", "#Raad"],
    }
    tags = " ".join(tags_map.get(dt, ["#DenHaag", "#Raad"]))
    # 1) Feitelijk
    t1 = f"{title} uit {mt} ({md}) is gepubliceerd. Lees het document: {url} {tags}"
    # 2) Urgent/duiding
    t2 = f"Nieuw: {dt} bij {mt} ({md}). Wat betekent dit voor Den Haag? Check het document: {url} {tags}"
    # 3) Luchtig/toegankelijk
    t3 = f"Vers van de pers uit de raad: {title}. Even bijlezen? {url} {tags}"
    # Truncate to 280 (X auto-shortens URLs maar we knippen toch defensief)
    def clamp(s): 
        return (s[:277] + '…') if len(s) > 280 else s
    return [clamp(t1), clamp(t2), clamp(t3)]

def post_tweets(tweets):
    """Post list of strings to X via Tweepy; or print in dry-run."""
    api_key = os.getenv("X_API_KEY")
    api_secret = os.getenv("X_API_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_secret = os.getenv("X_ACCESS_SECRET")
    if DRY_RUN or not all([api_key, api_secret, access_token, access_secret]):
        print("\n— DRY RUN — Tweets die zouden worden geplaatst:")
        for tw in tweets:
            print("-", tw)
        return

    import tweepy
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
    api = tweepy.API(auth)
    for tw in tweets:
        api.update_status(status=tw)
        time.sleep(2)  # heel klein beetje pauze

def main():
    if not META_CSV.exists():
        print(f"⚠️ Metadata CSV niet gevonden: {META_CSV}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(META_CSV)
    # Bepaal 'unieke' ID per document
    # Gebruik doc_id als primair, val terug op doc_url
    df["unique_id"] = df.apply(lambda r: str(r.get("doc_id")) if pd.notna(r.get("doc_id")) and str(r.get("doc_id")).strip() != "" else str(r.get("doc_url")), axis=1)
    seen = load_seen()
    new_df = df[~df["unique_id"].isin(seen)].copy()
    if new_df.empty:
        print("Geen nieuwe uploads sinds de vorige run.")
        return
    # Opbouw tweets per document
    all_tweets = []
    for _, row in new_df.iterrows():
        all_tweets.extend(mk_variants(row))
        seen.add(str(row["unique_id"]))
    # Bewaar nieuwe 'seen' set
    save_seen(seen)
    # Post of print
    post_tweets(all_tweets)
    print(f"✔ Klaar. {len(new_df)} nieuwe documenten; {len(all_tweets)} tweets voorbereid.")

if __name__ == "__main__":
    main()
