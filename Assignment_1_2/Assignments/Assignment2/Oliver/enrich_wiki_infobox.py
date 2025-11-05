#!/usr/bin/env python3
# enrich_wiki_infobox.py
# Read a GraphML of artists, fetch Wikipedia infoboxes, extract genres, write results,
# and produce stats + a top-15 genre histogram.

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import requests
import mwparserfromhell
import matplotlib.pyplot as plt
import pandas as pd
from requests.adapters import HTTPAdapter, Retry


WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "InfoboxGenreCollector/1.0 (educational; contact: example@example.com)"
CACHE_DIR = "cache_wikitext"  # on-disk cache for wikitext, speeds up repeated runs
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class InfoboxResult:
    title: str
    found: bool
    infobox_name: Optional[str]
    params: Dict[str, str]  # lowercase parameter -> text value (plain text)
    raw_wikitext: Optional[str]


def build_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)
    sess.headers["User-Agent"] = USER_AGENT
    return sess


def cache_path_for_title(title: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip())
    return os.path.join(CACHE_DIR, f"{safe}.txt")


def fetch_wikitext(session: requests.Session, title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (resolved_title, wikitext) or (None, None) if not found.
    Tries the exact title first; if missing or disambiguation, falls back to search.
    """
    # 1) Try exact
    resolved, text, is_disambig = try_get_wikitext_exact(session, title)
    if text and not is_disambig:
        return resolved, text

    # 2) Fallback search
    resolved2, text2 = search_and_get_wikitext(session, title)
    return resolved2, text2


def try_get_wikitext_exact(session: requests.Session, title: str) -> Tuple[Optional[str], Optional[str], bool]:
    # Cache first
    cp = cache_path_for_title(title)
    if os.path.exists(cp):
        with open(cp, "r", encoding="utf-8") as f:
            cached = f.read()
        return title, cached, False  # we can’t easily mark disambig from cache; treat as not disambig

    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions|pageprops",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None, None, False

    for _, page in pages.items():
        if "missing" in page:
            return None, None, False
        resolved_title = page.get("title")
        pageprops = page.get("pageprops", {})
        is_disambig = "disambiguation" in pageprops

        revisions = page.get("revisions")
        if not revisions:
            return resolved_title, None, is_disambig
        content = revisions[0].get("slots", {}).get("main", {}).get("*")
        if content:
            # Save cache
            with open(cp, "w", encoding="utf-8") as f:
                f.write(content)
            return resolved_title, content, is_disambig

    return None, None, False


def search_and_get_wikitext(session: requests.Session, query: str) -> Tuple[Optional[str], Optional[str]]:
    # Favor band/musical pages in search results
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 10,
        "srprop": "snippet"
    }
    r = session.get(WIKI_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None, None

    # Try to pick the most promising hit (band/musician/group)
    def score(hit):
        title = hit.get("title", "").lower()
        snippet = hit.get("snippet", "").lower()
        v = 0
        if any(k in title for k in ["(band)", "(musical group)", "(musician)"]):
            v += 3
        if "band" in snippet or "musician" in snippet or "singer" in snippet:
            v += 1
        return v

    hits_sorted = sorted(hits, key=score, reverse=True)
    for h in hits_sorted:
        t = h.get("title")
        resolved, text, is_disambig = try_get_wikitext_exact(session, t)
        if text and not is_disambig:
            return resolved, text

    # As last resort, return first even if disambig
    t = hits_sorted[0].get("title")
    resolved, text, _ = try_get_wikitext_exact(session, t)
    return resolved, text


def pick_infobox(wikitext: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Parse wikitext, pick the most relevant infobox, and return (infobox_name, {param->value_text}).
    Preference order:
      - infobox whose name contains 'musical artist' or 'musical group'
      - any infobox that contains a 'genre' parameter
      - otherwise the first infobox seen
    """
    code = mwparserfromhell.parse(wikitext)
    templates = [t for t in code.ifilter_templates(recursive=True) if "infobox" in t.name.lower()]
    if not templates:
        return None, {}

    def template_rank(t):
        name = t.name.strip_code().strip().lower()
        score = 0
        if "musical artist" in name or "musical group" in name:
            score += 10
        # Has genre parameter?
        for p in t.params:
            if p.name.strip().lower() == "genre":
                score += 3
                break
        return score

    templates_sorted = sorted(templates, key=template_rank, reverse=True)
    best = templates_sorted[0]
    name = best.name.strip_code().strip()

    params: Dict[str, str] = {}
    for p in best.params:
        k = str(p.name).strip().lower()
        v = best.get(p.name).value
        # Clean to plain text
        text = mwparserfromhell.parse(str(v)).strip_code(normalize=True, collapse=True)
        text = re.sub(r"\s+", " ", text).strip()
        params[k] = text

    return name, params


def extract_list_from_value_node(value_node) -> List[str]:
    """
    Pull a list of items out of a wikitext value node, respecting {{hlist|...}} and wikilinks when present.
    Fallback to delimiter-based splitting on commas, slashes, bullets and <br>.
    """
    out: List[str] = []
    val_code = mwparserfromhell.parse(str(value_node))

    # 1) Handle explicit list templates like {{hlist|...}} {{ubl|...}} {{flatlist|...}}
    for tmpl in val_code.ifilter_templates(recursive=True):
        name = tmpl.name.strip_code().strip().lower()
        if name in {"hlist", "ubl", "flatlist", "plainlist"}:
            # positional params 1..n
            for p in tmpl.params:
                if str(p.name).isdigit():
                    item = mwparserfromhell.parse(str(p.value)).strip_code(normalize=True, collapse=True)
                    item = item.strip()
                    if item:
                        out.append(item)
            if out:
                return out  # done

    # 2) If there are wikilinks, prefer their display text/title
    links = list(val_code.ifilter_wikilinks(recursive=True))
    if links:
        for link in links:
            text = (link.text or link.title).strip_code().strip()
            text = str(text)
            if text:
                out.append(text)
        if out:
            return out

    # 3) Fallback: plain text split
    plain = val_code.strip_code(normalize=True, collapse=True)
    plain = re.sub(r"(?i)<br\s*/?>", "|", plain)
    # Split on commas, pipes, bullets, slashes
    parts = re.split(r"\s*(?:,|/|•|·|\||;|\band\b)\s*", plain)
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def normalize_genre(g: str) -> str:
    s = g.strip().lower()
    # Remove parentheticals and references like [1]
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\[\d+\]", "", s)
    # Normalize punctuation/spaces
    s = s.replace("’", "'")
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Unify rock and roll variants
    s = re.sub(r"rock\s*(?:'?\s*n\s*'?\s*| and | & )roll", "rock and roll", s)
    # Remove trailing " music"
    s = re.sub(r"\s+music$", "", s)
    # Replace hyphen-like stuff with spaces (e.g., "heavy-metal")
    s = re.sub(r"[-–—]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_genres_from_wikitext(wikitext: str) -> List[str]:
    code = mwparserfromhell.parse(wikitext)
    # Find the chosen infobox; but for better extraction, we’ll read the value node directly
    chosen_name, _ = pick_infobox(wikitext)
    if not chosen_name:
        return []

    # Now find the chosen template again to access raw value nodes
    chosen = None
    for t in code.ifilter_templates(recursive=True):
        if t.name.strip_code().strip() == chosen_name:
            chosen = t
            break
    if not chosen:
        return []

    # Grab the "genre" parameter (if any)
    for p in chosen.params:
        if p.name.strip().lower() == "genre":
            items = extract_list_from_value_node(p.value)
            genres = [normalize_genre(x) for x in items]
            # Filter empties, dedupe preserving order
            seen = set()
            out: List[str] = []
            for g in genres:
                if not g:
                    continue
                if g not in seen:
                    seen.add(g)
                    out.append(g)
            return out
    return []


def get_infobox(session: requests.Session, title: str) -> InfoboxResult:
    resolved, wikitext = fetch_wikitext(session, title)
    if not wikitext or not resolved:
        return InfoboxResult(title=title, found=False, infobox_name=None, params={}, raw_wikitext=None)
    name, params = pick_infobox(wikitext)
    return InfoboxResult(title=resolved, found=bool(params), infobox_name=name, params=params, raw_wikitext=wikitext)


def main():
    ap = argparse.ArgumentParser(description="Enrich a GraphML of artists with Wikipedia infobox genres.")
    ap.add_argument("-i", "--input", required=True, help="Input GraphML")
    ap.add_argument("-o", "--output", default="rock_network_with_genres.graphml", help="Output GraphML with genre attributes")
    ap.add_argument("-j", "--json", default="genres.json", help="Output JSON mapping artist->genres")
    ap.add_argument("-c", "--csv", default="genres.csv", help="Output CSV with artist,genre")
    ap.add_argument("-p", "--plot", default="genre_histogram.png", help="Output PNG plot for top-15 genres")
    ap.add_argument("--delay", type=float, default=0.2, help="Politeness delay between requests (seconds)")
    args = ap.parse_args()

    session = build_session()

    print(f"Loading graph: {args.input}")
    G = nx.read_graphml(args.input)

    # Node names: assume node IDs are the Wikipedia titles
    node_titles = list(G.nodes())
    print(f"Found {len(node_titles)} nodes.")

    artist_to_genres: Dict[str, List[str]] = {}
    distinct_genres: Counter = Counter()
    got_count = 0

    for idx, node in enumerate(node_titles, start=1):
        title = str(node)
        try:
            info = get_infobox(session, title)
            if not info.found:
                print(f"[{idx}/{len(node_titles)}] {title} -> no infobox found.")
                continue

            genres = extract_genres_from_wikitext(info.raw_wikitext or "")
            if genres:
                artist_to_genres[title] = genres
                got_count += 1
                for g in genres:
                    distinct_genres[g] += 1

                # Store on node; GraphML wants strings
                G.nodes[node]["genres"] = "; ".join(genres)
                G.nodes[node]["genre_count"] = str(len(genres))
                # Optional: a few extra nice-to-haves if present
                for k_in, k_out in [("origin", "origin"), ("years_active", "years_active"), ("label", "labels")]:
                    if k_in in info.params and info.params[k_in]:
                        G.nodes[node][k_out] = info.params[k_in]
                print(f"[{idx}/{len(node_titles)}] {title} -> {len(genres)} genres")
            else:
                print(f"[{idx}/{len(node_titles)}] {title} -> genre not found.")
        except Exception as e:
            print(f"[{idx}/{len(node_titles)}] {title} -> ERROR: {e}")
        time.sleep(args.delay)

    # Save artifacts
    print(f"\nWriting updated GraphML -> {args.output}")
    nx.write_graphml(G, args.output)

    print(f"Writing JSON -> {args.json}")
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(artist_to_genres, f, ensure_ascii=False, indent=2)

    print(f"Writing CSV -> {args.csv}")
    # CSV: rows of artist, genre (one row per pair)
    rows = []
    for artist, genres in artist_to_genres.items():
        for g in genres:
            rows.append({"artist": artist, "genre": g})
    pd.DataFrame(rows).to_csv(args.csv, index=False)

    # Stats
    num_nodes = len(node_titles)
    num_with_genres = sum(1 for _ in artist_to_genres)
    total_genre_mentions = sum(len(v) for v in artist_to_genres.values())
    avg_genres_per_node = (total_genre_mentions / num_with_genres) if num_with_genres else 0.0
    total_distinct_genres = len(distinct_genres)

    print("\n=== Genre Coverage Stats ===")
    print(f"Nodes total:                     {num_nodes}")
    print(f"Nodes with genres found:         {num_with_genres}")
    print(f"Average genres per found node:   {avg_genres_per_node:.2f}")
    print(f"Distinct genres total:           {total_distinct_genres}")

    # Top-15 histogram
    top15 = distinct_genres.most_common(15)
    if top15:
        labels, counts = zip(*top15)
        plt.figure(figsize=(12, 6))
        plt.bar(labels, counts)
        plt.xticks(rotation=45, ha="right")
        plt.title("Top 15 Genres (by artist count)")
        plt.ylabel("Artist count")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=160)
        print(f"Saved top-15 histogram -> {args.plot}")
    else:
        print("No genres found; not generating histogram.")


if __name__ == "__main__":
    main()
