"""
Morning Brief pipeline orchestrator - runs the scout, analyst, and editor
stages, then renders and sends the final email.

Required environment variables:
    ANTHROPIC_API_KEY  - for all three model stages
    RESEND_API_KEY     - for the send stage only
    TO_EMAIL           - for the send stage only
    FROM_EMAIL         - optional for the send stage; defaults to Resend's
                          sandbox sender, which can only deliver to the
                          email address on the Resend account itself until
                          a custom domain is verified

Usage:
    python main.py                    # run the full pipeline
    python main.py --stage scout      # run just one stage, for debugging
    python main.py --stage analyst
    python main.py --stage editor
    python main.py --stage send       # renders + sends brief.json
"""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import anthropic
import requests

CANDIDATES_FILE = "candidates.csv"
PROFILE_FILE = "user_profile.json"
SEEN_FILE = "seen_articles.csv"
SCOUT_PROMPT_FILE = "prompts/scout_prompt.txt"
ANALYST_PROMPT_FILE = "prompts/analyst_prompt.txt"
EDITOR_PROMPT_FILE = "prompts/editor_prompt.txt"
SCOUT_OUTPUT_FILE = "scout_output.csv"
ANALYST_OUTPUT_FILE = "analyst_output.json"
BRIEF_FILE = "brief.json"

# Cost-optimized defaults: everything on Haiku. The plan's original design
# called for mid-tier/strongest models on analyst/editor, but at daily
# volume that lands well into hundreds of dollars/year - Sonnet or Opus are
# easy upgrades on the two lines below (each is a single string swap) once
# real usage numbers show there's budget for it. Editor is the stage where
# it'll show up most if you do upgrade one.
SCOUT_MODEL = "claude-haiku-4-5-20251001"
ANALYST_MODEL = "claude-haiku-4-5-20251001"  # upgrade option: "claude-sonnet-5"
EDITOR_MODEL = "claude-haiku-4-5-20251001"  # upgrade option: "claude-sonnet-5" or "claude-opus-5"

SCORE_THRESHOLD = 6
# Hard caps so a heavy news day can't blow the budget: only the top
# MAX_STORIES scouted items go to analyst/editor at all, and only the top
# FETCH_TOP_N of those get a full-text fetch attempt (the rest always get a
# snippet-only summary) - full-text fetches are the single biggest cost
# driver in the analyst stage.
MAX_STORIES = 20
FETCH_TOP_N = 10
# Smaller sub-batches for the analyst call so a cheap model has less to keep
# track of at once - a single 20-item batch was observed mixing up which
# sources/links belonged to which story near the end of the list.
ANALYST_BATCH_SIZE = 8
# How far back to look in seen_articles.csv when telling scout what's
# already been covered, so it can deprioritize "same story, no new
# developments" candidates instead of repeating them day after day.
RECENT_TOPICS_DAYS = 7
# Midpoint of the 200-225 wpm silent-reading range the editor prompt uses -
# read time is computed from this in code, never trusted from the model.
READING_WPM = 212


def hash_url(url):
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Scout stage
# ---------------------------------------------------------------------------

SCOUT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url_hashes": {"type": "array", "items": {"type": "string"}},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["url_hashes", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def load_candidates(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_recent_topics(path, days):
    """Headlines sent in the last `days` days, for scout's repetition check.
    Returns [] gracefully if the file doesn't exist yet or has no headline
    column (e.g. a fresh seen_articles.csv before the first real send)."""
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    topics = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                headline = row.get("headline", "").strip()
                if not headline:
                    continue
                try:
                    sent = datetime.strptime(row["date_sent"], "%Y-%m-%d").date()
                except (KeyError, ValueError):
                    continue
                if sent >= cutoff:
                    topics.append(headline)
    except FileNotFoundError:
        pass
    return sorted(set(topics))


def candidates_for_scout_prompt(candidates):
    # Trim to what the scout actually needs to score - keeps the request
    # small and avoids spending tokens on fields it doesn't use.
    return [
        {
            "url_hash": c["url_hash"],
            "source_id": c["ID"],
            "source": c["Source"],
            "category": c["Category"],
            "source_priority": c["Priority"],
            "title": c["title"],
            "published": c["published"],
            "summary": c["summary"][:400],
        }
        for c in candidates
    ]


def call_scout(client, candidates):
    with open(PROFILE_FILE, encoding="utf-8") as f:
        profile = json.load(f)
    with open(SCOUT_PROMPT_FILE, encoding="utf-8") as f:
        system_prompt = f.read()

    recent_topics = load_recent_topics(SEEN_FILE, RECENT_TOPICS_DAYS)
    user_content = json.dumps(
        {
            "user_profile": profile,
            "recently_covered": recent_topics,
            "candidates": candidates_for_scout_prompt(candidates),
        }
    )

    # One JSON item per cluster (url_hashes + score + reason) for ~350
    # candidates can run past 16k output tokens, which risks silent
    # truncation - stream with headroom instead of a small non-streaming cap.
    with client.messages.stream(
        model=SCOUT_MODEL,
        max_tokens=64000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        output_config={"format": {"type": "json_schema", "schema": SCOUT_OUTPUT_SCHEMA}},
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "max_tokens":
        print(
            "[WARN] scout response hit max_tokens - output is truncated/invalid JSON. "
            "Try splitting candidates.csv into smaller batches.",
            file=sys.stderr,
        )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["items"]


def write_scout_output(items, candidates, path):
    by_hash = {c["url_hash"]: c for c in candidates}

    passing = [i for i in items if i["score"] >= SCORE_THRESHOLD]
    passing.sort(key=lambda i: i["score"], reverse=True)
    passing = passing[:MAX_STORIES]  # hard cap regardless of how many scored high

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "score",
                "reason",
                "sources",
                "titles",
                "links",
                "url_hashes",
                "fetch_full_text",
            ],
        )
        writer.writeheader()
        for rank, item in enumerate(passing):
            members = [by_hash[h] for h in item["url_hashes"] if h in by_hash]
            if not members:
                continue
            writer.writerow(
                {
                    "score": item["score"],
                    "reason": item["reason"],
                    "sources": "; ".join(m["Source"] for m in members),
                    "titles": "; ".join(m["title"] for m in members),
                    "links": "; ".join(m["link"] for m in members),
                    "url_hashes": "; ".join(item["url_hashes"]),
                    "fetch_full_text": rank < FETCH_TOP_N,
                }
            )


def run_scout_stage(client):
    candidates = load_candidates(CANDIDATES_FILE)
    if not candidates:
        print(f"No candidates in {CANDIDATES_FILE} - run fetch_feeds.py first.")
        sys.exit(1)

    recent_count = len(load_recent_topics(SEEN_FILE, RECENT_TOPICS_DAYS))
    print(
        f"[scout] scoring {len(candidates)} candidate(s) against {PROFILE_FILE} "
        f"({recent_count} recently-covered headline(s) from the last {RECENT_TOPICS_DAYS} days)..."
    )
    items = call_scout(client, candidates)

    total_sources = sum(len(i["url_hashes"]) for i in items)
    passed = [i for i in items if i["score"] >= SCORE_THRESHOLD]
    write_scout_output(items, candidates, SCOUT_OUTPUT_FILE)

    kept = min(len(passed), MAX_STORIES)
    print(
        f"[scout] {len(items)} item(s) after clustering ({total_sources} source article(s) "
        f"considered), {len(passed)} scored >= {SCORE_THRESHOLD}, top {kept} kept "
        f"(top {min(kept, FETCH_TOP_N)} will get full-text fetches) -> {SCOUT_OUTPUT_FILE}"
    )


# ---------------------------------------------------------------------------
# Analyst stage
# ---------------------------------------------------------------------------

ANALYST_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "integer"},
                    "headline": {"type": "string"},
                    "summary": {"type": "string"},
                    "bias_label": {"type": "string"},
                    "headline_only": {"type": "boolean"},
                },
                "required": [
                    "item_id",
                    "headline",
                    "summary",
                    "bias_label",
                    "headline_only",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def load_candidates_by_hash(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["url_hash"]: row for row in csv.DictReader(f)}


def build_analyst_batch(scout_rows, candidates_by_hash):
    """One entry per scout row, with its member articles and whether this
    item is eligible for a full-text fetch (top FETCH_TOP_N only - keeps
    the single biggest cost driver bounded regardless of how many stories
    pass the scout threshold on a given day)."""
    batch = []
    lookup = {}
    for item_id, row in enumerate(scout_rows):
        hashes = [h.strip() for h in row["url_hashes"].split(";") if h.strip()]
        members = [candidates_by_hash[h] for h in hashes if h in candidates_by_hash]
        if not members:
            continue
        lookup[item_id] = members
        batch.append(
            {
                "item_id": item_id,
                "fetch_full_text": row["fetch_full_text"] in ("True", "true", "1"),
                "articles": [
                    {
                        "source": m["Source"],
                        "category": m["Category"],
                        "link": m["link"],
                        "published": m["published"],
                        "full_text_available": m["Full_Text"],
                        "feed_snippet": m["summary"],
                        "title": m["title"],
                    }
                    for m in members
                ],
            }
        )
    return batch, lookup


def call_analyst(client, system_prompt, batch):
    fetch_eligible = sum(1 for b in batch if b["fetch_full_text"])
    # Haiku doesn't support programmatic tool calling, which both web tools
    # otherwise default to allowing - restrict both to plain direct calls.
    tools = [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            # Bounded but generous - grounding lookups ("what is the Bosnian
            # genocide") should be occasional, not one per item.
            "max_uses": 6,
            "allowed_callers": ["direct"],
        }
    ]
    if fetch_eligible > 0:
        tools.append(
            {
                "type": "web_fetch_20260209",
                "name": "web_fetch",
                "max_uses": min(fetch_eligible * 2, 20),
                "max_content_tokens": 2000,  # bounds the cost of any single fetch
                "allowed_callers": ["direct"],
            }
        )

    with client.messages.stream(
        model=ANALYST_MODEL,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": json.dumps({"items": batch})}],
        tools=tools,
        output_config={"format": {"type": "json_schema", "schema": ANALYST_OUTPUT_SCHEMA}},
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "max_tokens":
        print(
            "[WARN] analyst response hit max_tokens - output may be truncated/invalid JSON",
            file=sys.stderr,
        )

    text_blocks = [b for b in response.content if b.type == "text"]
    text = text_blocks[-1].text if text_blocks else "{}"
    return json.loads(text).get("items", [])


def run_analyst_stage(client):
    with open(SCOUT_OUTPUT_FILE, newline="", encoding="utf-8") as f:
        scout_rows = list(csv.DictReader(f))
    if not scout_rows:
        print(f"No items in {SCOUT_OUTPUT_FILE} - run the scout stage first.")
        sys.exit(1)

    candidates_by_hash = load_candidates_by_hash(CANDIDATES_FILE)
    with open(ANALYST_PROMPT_FILE, encoding="utf-8") as f:
        system_prompt = f.read()

    batch, lookup = build_analyst_batch(scout_rows, candidates_by_hash)

    results = []
    for start in range(0, len(batch), ANALYST_BATCH_SIZE):
        chunk = batch[start : start + ANALYST_BATCH_SIZE]
        print(
            f"[analyst] batch {start // ANALYST_BATCH_SIZE + 1}: {len(chunk)} item(s) "
            f"({sum(1 for b in chunk if b['fetch_full_text'])} with full-text fetch)..."
        )
        # One retry before giving up on a batch - observed a batch fail with
        # invalid_request_error and no further detail, then succeed on an
        # identical retry, suggesting a transient upstream blip rather than
        # something wrong with the request itself.
        raw_results = None
        last_exc = None
        for attempt in range(2):
            try:
                raw_results = call_analyst(client, system_prompt, chunk)
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    print(
                        f"  [WARN] batch {start // ANALYST_BATCH_SIZE + 1} failed "
                        f"({exc}) - retrying once",
                        file=sys.stderr,
                    )
        if raw_results is None:
            headlines = [a["title"] for b in chunk for a in b["articles"]]
            print(
                f"  [FAIL] batch {start // ANALYST_BATCH_SIZE + 1} failed twice, "
                f"skipping {len(chunk)} item(s) ({headlines}): {last_exc}",
                file=sys.stderr,
            )
            continue

        seen_ids = set()
        for r in raw_results:
            item_id = r.get("item_id")
            if item_id in seen_ids:
                print(f"  [WARN] duplicate item_id {item_id} in response, skipping", file=sys.stderr)
                continue
            members = lookup.get(item_id)
            if not members:
                print(f"  [WARN] unknown item_id {item_id} in response, skipping", file=sys.stderr)
                continue
            seen_ids.add(item_id)
            r["sources"] = [m["Source"] for m in members]
            r["links"] = [m["link"] for m in members]
            r["category"] = members[0]["Category"]
            r["priority"] = max(int(m["Priority"]) for m in members)
            del r["item_id"]
            results.append(r)

        expected_ids = {b["item_id"] for b in chunk}
        missing = expected_ids - seen_ids
        if missing:
            print(f"  [WARN] analyst never returned item_id(s) {sorted(missing)}", file=sys.stderr)

    with open(ANALYST_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[analyst] {len(results)}/{len(batch)} item(s) analyzed -> {ANALYST_OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Editor stage
# ---------------------------------------------------------------------------

EDITOR_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_title": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_ids": {"type": "array", "items": {"type": "integer"}},
                                "headline": {"type": "string"},
                                "body": {"type": "string"},
                                "bias_label": {"type": "string"},
                            },
                            "required": ["item_ids", "headline", "body", "bias_label"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["section_title", "items"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["subject", "sections"],
    "additionalProperties": False,
}


def parse_target_word_count(briefing_length):
    """'600-1800 words' -> 'approximately 600-1800 words'. Falls back to a
    plain description if the preference string isn't in the expected shape,
    rather than crashing the whole stage over a formatting quirk. This is a
    range to land in given genuinely-sized items, not a floor to pad up to -
    a day with mostly thin/headline-only stories should sit near the low
    end, not be stretched."""
    match = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*words?", briefing_length, re.I)
    if not match:
        return briefing_length or "no specific target given"
    low, high = int(match.group(1)), int(match.group(2))
    return f"approximately {low}-{high} words"


def run_editor_stage(client):
    with open(ANALYST_OUTPUT_FILE, encoding="utf-8") as f:
        analyst_items = json.load(f)
    if not analyst_items:
        print(f"No items in {ANALYST_OUTPUT_FILE} - run the analyst stage first.")
        sys.exit(1)

    with open(PROFILE_FILE, encoding="utf-8") as f:
        profile = json.load(f)
    with open(EDITOR_PROMPT_FILE, encoding="utf-8") as f:
        system_prompt = f.read()

    # Editor references source articles by id and never has to retype a
    # link or source name itself - long opaque Google News URLs are exactly
    # the kind of string an LLM occasionally mistypes by one character when
    # asked to reproduce them, which produces a broken "click here" link
    # with no error anywhere. Code resolves ids to the real values instead.
    editor_articles = [
        {
            "id": i,
            "headline": a["headline"],
            "summary": a["summary"],
            "bias_label": a["bias_label"],
            "headline_only": a["headline_only"],
            "category": a["category"],
            "priority": a["priority"],
        }
        for i, a in enumerate(analyst_items)
    ]

    target_words = parse_target_word_count(
        profile.get("preferences", {}).get("briefing_length", "")
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_content = json.dumps(
        {
            "user_profile": profile,
            "date": today,
            "target_word_count": target_words,
            "articles": editor_articles,
        }
    )

    print(f"[editor] composing brief from {len(analyst_items)} analyzed item(s)...")

    with client.messages.stream(
        model=EDITOR_MODEL,
        max_tokens=32000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        output_config={"format": {"type": "json_schema", "schema": EDITOR_OUTPUT_SCHEMA}},
    ) as stream:
        response = stream.get_final_message()

    text = next(b.text for b in response.content if b.type == "text")
    brief = json.loads(text)

    # Resolve sources/links, and work out which section each item actually
    # belongs in. Placement is only genuinely ambiguous for AI-category
    # items (that's the whole point of asking the editor to fold them in) -
    # a native UK/Global Politics story already has an unambiguous home, so
    # don't leave that to model judgement: override the editor's section
    # choice whenever every item_id it merged agrees on a non-AI category.
    allowed_sections = {"UK Politics", "Global Politics"}
    regrouped = {name: [] for name in allowed_sections}
    used_ids = set()
    for section in brief["sections"]:
        for item in section["items"]:
            sources, links, native_cats = [], [], set()
            for item_id in item["item_ids"]:
                if not (0 <= item_id < len(analyst_items)):
                    print(f"[editor] [WARN] out-of-range item_id {item_id}, skipping", file=sys.stderr)
                    continue
                used_ids.add(item_id)
                src = analyst_items[item_id]
                if src["category"] in allowed_sections:
                    native_cats.add(src["category"])
                for s, l in zip(src["sources"], src["links"]):
                    if (s, l) not in zip(sources, links):
                        sources.append(s)
                        links.append(l)
            item["sources"] = sources
            item["links"] = links
            del item["item_ids"]

            target = section["section_title"]
            if len(native_cats) == 1:
                (only_cat,) = native_cats
                if only_cat != target:
                    print(
                        f"[editor] [WARN] recategorizing '{item['headline'][:60]}' "
                        f"from {target} to {only_cat}",
                        file=sys.stderr,
                    )
                target = only_cat
            if target in allowed_sections and item["sources"]:
                regrouped[target].append(item)

    # Every analyzed item should end up in the brief somewhere (as its own
    # item, or folded into a merge) - there's no length budget to cut for
    # any more. Rather than just warn when the editor drops one anyway,
    # guarantee it: synthesize a minimal item straight from the analyst's
    # own data and insert it. A plain, unedited item reaching Ned beats one
    # that silently never existed.
    missing = set(range(len(analyst_items))) - used_ids
    for item_id in sorted(missing):
        src = analyst_items[item_id]
        print(
            f"[editor] [WARN] '{src['headline'][:60]}' missing from the brief - "
            f"inserting unedited",
            file=sys.stderr,
        )
        target = src["category"] if src["category"] in allowed_sections else "Global Politics"
        regrouped[target].append(
            {
                "headline": src["headline"],
                "body": src["summary"],
                "bias_label": src["bias_label"],
                "sources": src["sources"],
                "links": src["links"],
            }
        )

    brief["sections"] = [
        {"section_title": name, "items": regrouped[name]}
        for name in ("UK Politics", "Global Politics")
        if regrouped[name]
    ]

    # Read time is measured from actual output, never asked of the model -
    # "how many words did I just write" turned out to be something the
    # model could not estimate at all (observed off by 5-10x, consistently
    # reporting the target it was told to aim for rather than what it
    # actually wrote). READING_WPM is the midpoint of the 200-225 wpm range
    # the prompt specifies.
    word_count = sum(
        len(item["body"].split()) for s in brief["sections"] for item in s["items"]
    )
    brief["estimated_read_minutes"] = max(1, round(word_count / READING_WPM))

    with open(BRIEF_FILE, "w", encoding="utf-8") as f:
        json.dump(brief, f, indent=2)

    n_items = sum(len(s["items"]) for s in brief["sections"])
    print(
        f"[editor] '{brief['subject']}' - {len(brief['sections'])} section(s), "
        f"{n_items} item(s), {word_count} words, ~{brief['estimated_read_minutes']} "
        f"min (measured) -> {BRIEF_FILE}"
    )


# ---------------------------------------------------------------------------
# Render + send stage
# ---------------------------------------------------------------------------

def render_html(brief):
    # Matches the masthead/wire-dispatch design reviewed in chat, but rebuilt
    # email-safe: every rule is inline (no CSS custom properties / :root -
    # Outlook and other clients don't reliably apply those), no flex/grid,
    # and no dark-mode media query, since dark-mode email support is patchy
    # enough that one well-designed light theme is the safer default. The
    # Google Fonts link degrades gracefully to the serif/monospace fallback
    # stack on clients that strip external font loading.
    INK = "#1C2333"
    INK_MUTED = "#5A6275"
    ACCENT = "#A5690E"
    ACCENT_SOFT = "#C98A2E"
    RULE = "#DAD8E4"
    BG = "#F2F1F6"
    SERIF = "'Newsreader', Georgia, 'Times New Roman', serif"
    MONO = "'Space Mono', 'Courier New', monospace"

    # Avoid strftime's "%-d" (no leading zero) - it's a Linux-only extension
    # that raises on Windows, and this runs both locally and on GH Actions.
    now = datetime.now(timezone.utc)
    today_display = f"{now.day} {now.strftime('%B %Y')}"

    parts = [
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;'
        '0,500;0,600;1,500;1,600&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">',
        f'<div style="max-width:680px;margin:0 auto;padding:32px 20px;background:{BG};'
        f'font-family:{SERIF};color:{INK};-webkit-font-smoothing:antialiased;">',
        f'<div style="text-align:center;border-top:2px solid {INK};'
        f'border-bottom:1px solid {INK};padding:20px 0 16px;margin-bottom:36px;">',
        f'<p style="font-family:{MONO};font-size:11px;letter-spacing:0.14em;'
        f'text-transform:uppercase;color:{INK_MUTED};margin:0 0 10px;">'
        f'Personal Briefing &middot; Wire &amp; Desk Reports</p>',
        f'<h1 style="font-family:{SERIF};font-style:italic;font-size:26px;'
        f'font-weight:600;line-height:1.28;margin:0 0 14px;">'
        f'{html.escape(brief["subject"])}</h1>',
        f'<p style="font-family:{MONO};font-size:11px;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{INK_MUTED};margin:0;">'
        f'{today_display}<span style="color:{RULE};margin:0 8px;">&bull;</span>'
        f'About {brief["estimated_read_minutes"]} minute read</p>',
        "</div>",
    ]

    for section in brief["sections"]:
        parts.append(
            f'<h2 style="font-family:{MONO};font-size:12px;font-weight:700;'
            f'letter-spacing:0.16em;text-transform:uppercase;color:{ACCENT};'
            f'border-bottom:1px solid {RULE};padding-bottom:10px;margin:40px 0 24px;">'
            f'{html.escape(section["section_title"])}</h2>'
        )
        items = section["items"]
        for idx, item in enumerate(items):
            border = f"border-bottom:1px solid {RULE};" if idx < len(items) - 1 else ""
            parts.append(f'<div style="padding:0 0 24px;margin-bottom:24px;{border}">')
            parts.append(
                f'<h3 style="font-family:{SERIF};font-size:19px;font-weight:600;'
                f'line-height:1.32;margin:0 0 8px;">{html.escape(item["headline"])}</h3>'
            )
            if item.get("bias_label"):
                parts.append(
                    f'<p style="font-family:{MONO};font-size:11px;color:{ACCENT_SOFT};'
                    f'margin:0 0 12px;">{html.escape(item["bias_label"])}</p>'
                )
            for para in item["body"].split("\n\n"):
                para_html = html.escape(para).replace("\n", "<br>")
                parts.append(
                    f'<p style="font-size:16px;line-height:1.6;margin:0 0 12px;'
                    f'max-width:62ch;">{para_html}</p>'
                )
            links_html = '<span style="color:{0};margin:0 6px;">/</span>'.format(RULE).join(
                f'<a href="{html.escape(l)}" style="color:{INK_MUTED};text-decoration:underline;">'
                f'{html.escape(s)}</a>'
                for s, l in zip(item["sources"], item["links"])
            )
            parts.append(
                f'<p style="font-family:{MONO};font-size:11px;color:{INK_MUTED};margin:0;">'
                f'SOURCES &mdash; {links_html}</p>'
            )
            parts.append("</div>")

    parts.append(
        f'<p style="font-family:{MONO};font-size:10px;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{INK_MUTED};text-align:center;'
        f'border-top:1px solid {RULE};padding-top:20px;margin-top:24px;">'
        f'The Morning Brief &middot; Compiled for Ned</p>'
    )
    parts.append("</div>")
    return "\n".join(parts)


def archive_and_mark_seen(brief):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs("output", exist_ok=True)
    html_content = render_html(brief)

    with open(f"output/{today}.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    with open(SEEN_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for section in brief["sections"]:
            for item in section["items"]:
                for link in item["links"]:
                    writer.writerow(
                        [hash_url(link), today, section["section_title"], item["headline"]]
                    )

    return html_content


def send_email(brief_html, subject):
    api_key = os.environ["RESEND_API_KEY"]
    # .strip() defensively - a GitHub secret pasted with a trailing newline
    # or leading space is a common, invisible cause of exactly this kind of
    # rejection, and there's no way to inspect a secret's raw value to rule
    # it out other than stripping it and logging what was actually used.
    to_email = os.environ["TO_EMAIL"].strip()
    # `.get(key, default)` only falls back when the key is fully absent - it
    # doesn't help when the key exists but is empty, which is exactly what
    # happens here: the workflow YAML references ${{ secrets.FROM_EMAIL }},
    # so GitHub Actions sets FROM_EMAIL="" whenever that secret doesn't
    # exist, rather than omitting the env var. `or` catches both cases.
    from_email = (os.environ.get("FROM_EMAIL") or "Morning Brief <onboarding@resend.dev>").strip()
    print(f"[send] from={from_email!r} to={to_email!r}")

    response = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"from": from_email, "to": [to_email], "subject": subject, "html": brief_html},
        timeout=30,
    )
    if not response.ok:
        # raise_for_status() alone only reports the status code - Resend's
        # response body explains *why* (unverified sender, bad recipient,
        # etc.), and that's the part actually worth seeing when this fails.
        print(f"[send] [FAIL] Resend {response.status_code}: {response.text}", file=sys.stderr)
    response.raise_for_status()
    return response.json()


def run_send_stage():
    with open(BRIEF_FILE, encoding="utf-8") as f:
        brief = json.load(f)

    print(f"[send] archiving and marking {BRIEF_FILE}'s links as seen...")
    brief_html = archive_and_mark_seen(brief)

    print(f"[send] sending '{brief['subject']}' to {os.environ.get('TO_EMAIL', '?')}...")
    result = send_email(brief_html, brief["subject"])
    print(f"[send] sent - Resend id: {result.get('id', '?')}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["scout", "analyst", "editor", "send", "all"],
        default="all",
        help="which pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    # Some accounts issue "identity-linked" API keys that require the target
    # workspace to be sent explicitly - harmless to set unconditionally.
    client_kwargs = {}
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace_id:
        client_kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
    client = anthropic.Anthropic(**client_kwargs)

    if args.stage in ("scout", "all"):
        run_scout_stage(client)
    if args.stage in ("analyst", "all"):
        run_analyst_stage(client)
    if args.stage in ("editor", "all"):
        run_editor_stage(client)
    if args.stage in ("send", "all"):
        run_send_stage()


if __name__ == "__main__":
    main()
