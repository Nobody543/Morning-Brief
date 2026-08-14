"""
Step 2 of the Morning Brief pipeline: pull every feed in sources.csv,
normalize entries, drop anything already in seen_articles.csv, and write
the rest to candidates.csv for the scout stage.

Usage:
    python fetch_feeds.py                  # daily-cadence sources only
    python fetch_feeds.py --include-weekly # also fetch weekly-cadence sources
"""

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests

SOURCES_FILE = "sources.csv"
SEEN_FILE = "seen_articles.csv"
CANDIDATES_FILE = "candidates.csv"

# A generic script UA gets 403'd by several sources (Chatham House, jobs.ac.uk,
# nature.com career feeds all did during scaffolding). A browser-like UA is
# what got most of them to work during manual spot-checks.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 MorningBrief/0.1"
)
REQUEST_TIMEOUT = 15


def load_sources(path, include_weekly):
    sources = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("Feed_URL", "").strip():
                continue
            cadence = row.get("Cadence", "daily").strip().lower()
            if cadence == "weekly" and not include_weekly:
                continue
            sources.append(row)
    return sources


def load_seen_hashes(path):
    seen = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url_hash = row.get("url_hash", "").strip()
                if url_hash:
                    seen.add(url_hash)
    except FileNotFoundError:
        pass
    return seen


def hash_url(url):
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def entry_published(entry):
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
    return None


def entry_summary(entry):
    summary = entry.get("summary", "") or entry.get("description", "")
    return " ".join(summary.split())


def fetch_feed(url):
    """Returns a feedparser result, raising on HTTP-level failures.
    feedparser itself never raises on malformed XML, it just reports it
    on the result's `bozo` flag, which callers should check."""
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return feedparser.parse(response.content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-weekly",
        action="store_true",
        help="also fetch weekly-cadence sources (normally run on a separate schedule)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help=(
            "only keep entries published within this many days (default: 2). "
            "Some feeds (e.g. OpenAI, Hugging Face) return their full history "
            "rather than just recent posts, so this keeps candidates.csv sane. "
            "Entries with no parseable date are kept regardless."
        ),
    )
    args = parser.parse_args()

    sources = load_sources(SOURCES_FILE, args.include_weekly)
    seen_hashes = load_seen_hashes(SEEN_FILE)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    candidates = []
    ok_count = 0
    fail_count = 0
    stale_skipped = 0

    for source in sources:
        source_id = source["ID"]
        feed_url = source["Feed_URL"]
        try:
            parsed = fetch_feed(feed_url)
        except requests.RequestException as exc:
            print(f"[FAIL] {source_id} ({source['Source']}): {exc}", file=sys.stderr)
            fail_count += 1
            continue

        if parsed.bozo and not parsed.entries:
            print(
                f"[FAIL] {source_id} ({source['Source']}): "
                f"unparseable feed - {parsed.bozo_exception}",
                file=sys.stderr,
            )
            fail_count += 1
            continue

        new_for_source = 0
        for entry in parsed.entries:
            link = entry.get("link", "").strip()
            if not link:
                continue
            url_hash = hash_url(link)
            if url_hash in seen_hashes:
                continue
            published = entry_published(entry)
            if published is not None and published < cutoff:
                stale_skipped += 1
                continue
            candidates.append(
                {
                    "url_hash": url_hash,
                    "ID": source_id,
                    "Source": source["Source"],
                    "Category": source["Category"],
                    "Priority": source["Priority"],
                    "Full_Text": source["Full_Text"],
                    "title": entry.get("title", "").strip(),
                    "link": link,
                    "published": published.isoformat() if published else "",
                    "summary": entry_summary(entry),
                }
            )
            new_for_source += 1

        print(f"[OK]   {source_id} ({source['Source']}): {new_for_source} new item(s)")
        ok_count += 1
        time.sleep(0.5)  # go easy on smaller sites

    with open(CANDIDATES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url_hash",
                "ID",
                "Source",
                "Category",
                "Priority",
                "Full_Text",
                "title",
                "link",
                "published",
                "summary",
            ],
        )
        writer.writeheader()
        writer.writerows(candidates)

    print(
        f"\n{ok_count} feed(s) OK, {fail_count} failed, "
        f"{stale_skipped} skipped as older than {args.days} day(s), "
        f"{len(candidates)} new candidate(s) written to {CANDIDATES_FILE}"
    )


if __name__ == "__main__":
    main()
