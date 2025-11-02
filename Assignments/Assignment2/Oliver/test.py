import re
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mwparserfromhell

CACHE_DIR = "cache_wikitext"   # same as in your enrichment script
LABMT_PATH = "pone.0026752.s001.tsv"
GRAPH_IN = "rock_network_with_genres.graphml"
GRAPH_OUT = "rock_network_with_sentiment.graphml"


# ---------------------------
# Load LabMT
# ---------------------------

def load_labmt(filepath: str) -> dict:
    labmt = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            word = parts[0].lower()
            try:
                score = float(parts[2])
            except ValueError:
                continue
            labmt[word] = score
    return labmt


def sentiment_score(tokens: list, labmt: dict,
                    drop_neutral: bool = True,
                    neutral_low: float = 4.0, neutral_high: float = 6.0) -> float:
    scores = []
    for t in tokens:
        if t in labmt:
            s = labmt[t]
            if drop_neutral and neutral_low <= s <= neutral_high:
                continue
            scores.append(s)
    if not scores:
        return np.nan
    return float(np.mean(scores))


def cache_path_for_title(title: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip())
    return os.path.join(CACHE_DIR, f"{safe}.txt")


# ---------------------------
# Main sentiment assignment
# ---------------------------

print("Loading graph...")
G = nx.read_graphml(GRAPH_IN)

print("Loading LabMT...")
labmt = load_labmt(LABMT_PATH)

node_sentiments = {}

for node in G.nodes():
    title = str(node)
    cache_file = cache_path_for_title(title)

    if not os.path.exists(cache_file):
        node_sentiments[node] = np.nan
        G.nodes[node]["sentiment"] = np.nan
        continue

    with open(cache_file, "r", encoding="utf-8") as f:
        wikitext = f.read()

    plain = mwparserfromhell.parse(wikitext).strip_code()
    tokens = re.findall(r"[a-zA-Z']+", plain.lower())

    score = sentiment_score(tokens, labmt, drop_neutral=True)
    node_sentiments[node] = score
    G.nodes[node]["sentiment"] = score


# ---------------------------
# Stats
# ---------------------------

sentiments = np.array([v for v in node_sentiments.values() if not np.isnan(v)])

print(f"Found {len(sentiments)} pages with sentiment")

mean_s = np.nanmean(sentiments)
median_s = np.nanmedian(sentiments)
var_s = np.nanvar(sentiments, ddof=1)
q25 = np.percentile(sentiments, 25)
q75 = np.percentile(sentiments, 75)

print("\nSentiment Stats:")
print(f"Mean: {mean_s:.3f}")
print(f"Median: {median_s:.3f}")
print(f"Variance: {var_s:.3f}")
print(f"25th percentile: {q25:.3f}")
print(f"75th percentile: {q75:.3f}")


# ---------------------------
# Histogram
# ---------------------------

plt.figure(figsize=(10,6))
plt.hist(sentiments, bins=30, edgecolor="black", alpha=0.7)
plt.axvline(mean_s, color="red", linestyle="--", label=f"Mean = {mean_s:.2f}")
plt.axvline(median_s, color="green", linestyle="--", label=f"Median = {median_s:.2f}")
plt.axvline(q25, color="purple", linestyle=":", label=f"25% = {q25:.2f}")
plt.axvline(q75, color="purple", linestyle=":", label=f"75% = {q75:.2f}")
plt.title("Wikipedia Rock Artist Page Sentiment (LabMT)")
plt.xlabel("Sentiment score")
plt.ylabel("Number of artists")
plt.legend()
plt.tight_layout()
plt.savefig("sentiment_histogram.png", dpi=150)
plt.show()


# ---------------------------
# Top happiest & saddest
# ---------------------------

artist_scores = [(n, s) for n, s in node_sentiments.items() if not np.isnan(s)]
artist_scores_sorted = sorted(artist_scores, key=lambda x: x[1], reverse=True)

print("\n🎉 Top 10 happiest artists:")
for a, s in artist_scores_sorted[:10]:
    print(f"{a}: {s:.3f}")

print("\n💀 Top 10 saddest artists:")
for a, s in artist_scores_sorted[-10:]:
    print(f"{a}: {s:.3f}")


# ---------------------------
# Save updated graph
# ---------------------------

nx.write_graphml(G, GRAPH_OUT)
print(f"\nSaved graph with sentiment → {GRAPH_OUT}")
