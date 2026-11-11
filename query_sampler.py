#!/usr/bin/env python3
"""
query_sampler.py
Builds the stratified query set for the AI answer engine attribution study.

Pipeline (four subcommands, each resumable and seeded):

  1. pool         Pull queries from public datasets and/or your own CSV files.
  2. label        Label each query with domain, intent and exclusion flags
                  (free local heuristic, or an LLM if you have an API key).
  3. sample       Filter, de-duplicate and sample N queries per domain x intent cell.
  5. review-make / review-apply
                  Hand-verify every final label, drop unusable queries, rebuild the set.
  4. audit-make / audit-score
                  Blind hand-check of the LLM labels, with accuracy and Cohen's kappa.

Example:
  pip install datasets sentence-transformers      # free path, no API key
  # LLM path instead: pip install datasets anthropic; export ANTHROPIC_API_KEY=...
  python query_sampler.py pool --nq 4000 --msmarco 4000 --extra autocomplete.csv --out pool.jsonl
  python query_sampler.py label --pool pool.jsonl --out labeled.jsonl --labeler heuristic
  python query_sampler.py sample --labeled labeled.jsonl --per-cell 4 --reserve 2 --out queries_final.csv
  python query_sampler.py audit-make --queries queries_final.csv --out audit_blind.csv
  # fill human_domain and human_intent in audit_blind.csv, then:
  python query_sampler.py audit-score --audit audit_blind.csv --queries queries_final.csv

Extra CSV files need a `query` column and optionally a `source` column
(use this for autocomplete / People Also Ask expansions).

Edit DOMAINS and INTENTS below to change the grid. Both lists drive the labeler
prompt, the sampler and the audit, so nothing else needs to change.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DOMAINS = [
    "health",
    "personal_finance",
    "consumer_tech",
    "science",
    "law_government",
    "travel",
    "food_cooking",
    "history_education",
    "home_diy",
    "consumer_products",
]

INTENTS = [
    "factual_lookup",
    "how_to",
    "comparison",
    "explanation",
    "recommendation",
]

FLAGS = [
    "navigational",
    "ambiguous",
    "time_sensitive",
    "unsafe_or_sensitive",
    "non_english",
    "off_topic",
]

# Verify these dataset IDs on Hugging Face before running; they can change.
NQ_ID = "sentence-transformers/natural-questions"
MSMARCO_ID, MSMARCO_CONFIG = "microsoft/ms_marco", "v2.1"

DEFAULT_MODEL = "claude-haiku-5-5"
MIN_WORDS, MAX_WORDS = 3, 20
NEARDUP_JACCARD = 0.7

URL_RE = re.compile(r"https?://|www\.|\.com\b|\.org\b|\.net\b", re.I)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def clean(q):
    return re.sub(r"\s+", " ", str(q)).strip()


def qid_of(q):
    return hashlib.sha1(clean(q).lower().encode("utf-8")).hexdigest()[:12]


def tokens(q):
    return set(re.findall(r"[a-z0-9]+", q.lower()))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def prefilter(q):
    """Cheap rule-based filter before any LLM spend."""
    n = len(q.split())
    if n < MIN_WORDS or n > MAX_WORDS:
        return False
    if URL_RE.search(q):
        return False
    body = q.replace(" ", "")
    if not body:
        return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in body)
    return ascii_letters / len(body) >= 0.8


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# --------------------------------------------------------------------------- #
# 1. pool
# --------------------------------------------------------------------------- #

def load_hf(dataset_id, config, n, seed, column="query"):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency: pip install datasets")
    kwargs = dict(split="train", streaming=True)
    ds = load_dataset(dataset_id, config, **kwargs) if config else load_dataset(dataset_id, **kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    out = []
    for row in ds:
        out.append(row[column])
        if len(out) >= n:
            break
    return out


def cmd_pool(a):
    rng = random.Random(a.seed)
    pool = {}
    raw_counts = Counter()
    kept_counts = Counter()

    def add(q, source):
        raw_counts[source] += 1
        q = clean(q)
        if not q or not prefilter(q):
            return
        qid = qid_of(q)
        if qid not in pool:
            pool[qid] = {"id": qid, "query": q, "source": source}
            kept_counts[source] += 1

    if a.nq > 0:
        for q in load_hf(NQ_ID, None, a.nq, a.seed):
            add(q, "natural_questions")
    if a.msmarco > 0:
        for q in load_hf(MSMARCO_ID, MSMARCO_CONFIG, a.msmarco, a.seed):
            add(q, "ms_marco")
    for path in a.extra or []:
        default_source = Path(path).stem
        for row in read_csv(path):
            add(row.get("query", ""), row.get("source") or default_source)

    rows = list(pool.values())
    rng.shuffle(rows)  # so --limit on the label step still gives a mixed sample
    Path(a.out).write_text("", encoding="utf-8")
    append_jsonl(a.out, rows)
    print(f"Pool written to {a.out}: {len(rows)} queries after filtering and de-duplication")
    for s in raw_counts:
        print(f"  {s}: {raw_counts[s]} raw -> {kept_counts[s]} kept")
    # Streaming datasets can leave background download threads retrying after we have
    # what we need. The pool file is already written and closed, so exit immediately.
    sys.stdout.flush()
    os._exit(0)


# --------------------------------------------------------------------------- #
# 2. label
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = f"""You label web search queries for a research study on AI answer engines.

For each query return:
- domain: one of {DOMAINS}, or "other" if none fits.
- intent: one of {INTENTS}, or "other" if none fits.
- flags: a list containing any of {FLAGS}; use [] if none apply.

Intent definitions:
- factual_lookup: asks for one specific fact (a date, number, name, place).
- how_to: asks for steps or a method to do something.
- comparison: asks to choose between named options, or "X vs Y".
- explanation: asks why or how something works, or what something means.
- recommendation: asks for the best or a suggested option.

Flag definitions:
- navigational: looking for a specific website or brand page.
- ambiguous: you cannot tell what is being asked.
- time_sensitive: the answer changes day to day (prices, scores, news, "latest", "current").
- unsafe_or_sensitive: self-harm, medical emergencies, explicit or hateful content.
- non_english: not primarily English.
- off_topic: about games, anime, entertainment or celebrities, not an everyday information need.

Return ONLY a JSON array, no prose and no code fences:
[{{"i": 0, "domain": "...", "intent": "...", "flags": []}}, ...]"""


def parse_json_array(text):
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in response")
    return json.loads(text[start:end + 1])


def normalise_label(item):
    domain = item.get("domain", "other")
    intent = item.get("intent", "other")
    flags = [f for f in item.get("flags", []) if f in FLAGS]
    return {
        "domain": domain if domain in DOMAINS else "other",
        "intent": intent if intent in INTENTS else "other",
        "flags": flags,
    }


def llm_labeler(model):
    try:
        import anthropic
    except ImportError:
        sys.exit("Missing dependency: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY, or use --labeler heuristic for the free option.")
    client = anthropic.Anthropic()

    def label(batch):
        payload = json.dumps([{"i": i, "q": r["query"]} for i, r in enumerate(batch)])
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": payload}],
                )
                items = parse_json_array(resp.content[0].text)
                return {int(it["i"]): normalise_label(it) for it in items if "i" in it}
            except Exception as e:  # retry on API errors and malformed output
                wait = 2 ** attempt
                print(f"  batch failed ({e}); retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
        return {}

    return label


def mock_labeler(batch):
    """Deterministic fake labels. For testing the pipeline only, never for real data."""
    out = {}
    for i, r in enumerate(batch):
        h = int(r["id"], 16)
        flags = ["time_sensitive"] if (h // 7) % 10 == 0 else []
        out[i] = {
            "domain": DOMAINS[h % len(DOMAINS)],
            "intent": INTENTS[(h // 13) % len(INTENTS)],
            "flags": flags,
        }
    return out


# ---- Free labeler: keyword rules for intent/flags, embeddings for domain ---- #

DOMAIN_DESCRIPTIONS = {
    "health": "health, medicine, diseases, symptoms, nutrition, fitness, mental health, medication",
    "personal_finance": "personal finance, banking, loans, taxes, investing, insurance, credit, budgeting",
    "consumer_tech": "smartphones, laptops, software, apps, the internet, gadgets, computers, how technology works",
    "science": "physics, chemistry, biology, astronomy, earth science, space, animals, how the natural world works",
    "law_government": "laws, courts, government, politics, rights, legal procedures, regulations",
    "travel": "travel, destinations, flights, hotels, visas, tourism, transportation, trip planning",
    "food_cooking": "cooking, recipes, ingredients, baking, restaurants, food, drinks",
    "history_education": "history, historical events, school, university, learning, education, geography, famous people",
    "home_diy": "home repair, DIY, gardening, furniture, plumbing, cleaning, appliances, home improvement",
    "consumer_products": "buying products, shopping, product reviews, household goods, clothing, cars, brands",
}

INTENT_RULES = [  # first match wins, so order matters
    ("explanation", re.compile(r"\bbest (describes|defines|explains|represents)\b", re.I)),
    ("factual_lookup", re.compile(r"^(does|did)\b.*\brecommend", re.I)),
    ("comparison", re.compile(r"\b(vs\.?|versus|compare[ds]?|comparison|difference between|better than|which is better|which one)\b|\b(best|better|worse|which|who)\b[^?]*\bor\b", re.I)),
    ("recommendation", re.compile(r"\b(best|top \d*|recommend\w*|should i (buy|get|use|choose)|good .* for)\b|^which\b[^?]*\b(to|should)\b", re.I)),
    ("how_to", re.compile(r"\b(how (to|do i|can i|should i|do you|can you)|steps to|ways to|guide to|tutorial)\b", re.I)),
    ("explanation", re.compile(r"\b(why|how does|how do .* work|what causes|what makes|explain|meaning of|what does .* mean)\b", re.I)),
    ("factual_lookup", re.compile(r"^(who|when|where|what|which|how (many|much|old|long|far|tall|big|fast|high))\b", re.I)),
]

FLAG_RULES = {
    "time_sensitive": re.compile(r"\b(today|tonight|right now|now|latest|current|currently|this (week|month|year)|news|score|scores|price of|stock|weather|live|20(2[3-9]))\b", re.I),
    "navigational": re.compile(r"\b(login|log in|sign in|website|official site|homepage|download|near me|customer service|phone number)\b", re.I),
    "unsafe_or_sensitive": re.compile(r"\b(suicide|kill myself|self.?harm|overdose|porn\w*|nude|rape|bomb|terroris\w*)\b", re.I),
    "off_topic": re.compile(r"\b(pokemon|pok\u00e9mon|pikachu|solgaleo|lunala|cosmoem|minecraft|fortnite|roblox|genshin|zelda|anime|manga|lyrics|walkthrough|cheat codes?)\b", re.I),
    # Malformed or fragmentary queries that an engine could not answer meaningfully.
    "ambiguous": re.compile(
        r"^how to \w+ work\b"
        r"|\bwhat does (do|does|is|are)\b"
        r"|\b(will|does|is|are|do|have|has|can|should)\s+vs\b"
        r"|\b(the|a|an|of|to|for|in|on|and|or|is|are|vs)\s*\??$",
        re.I,
    ),
}


def rule_intent(q):
    for name, rx in INTENT_RULES:
        if rx.search(q):
            return name
    return "other"


def rule_flags(q):
    return [name for name, rx in FLAG_RULES.items() if rx.search(q)]


DOMAIN_OVERRIDES = [  # keyword tie-breaks applied before the embedding match; first match wins
    ("personal_finance", re.compile(r"\binsurance\b|\bmortgages?\b|\btax(es)?\b|\brefunds?\b|\bloans?\b|\bcredit (score|card)s?\b|\b401k\b|\binvest\w*\b|\bretirement\b", re.I)),
    ("consumer_tech", re.compile(r"\b(iphone|android|samsung|pixel|wifi|wi-fi|router|laptop|macbook|chromebook|bluetooth|google maps)\b", re.I)),
    ("home_diy", re.compile(r"\b(paint\w*|drywall|faucet|lawn|compost\w*|plumb\w*|heat pump)\b", re.I)),
    ("food_cooking", re.compile(r"\b(air fryer|sourdough|cast iron|recipes?|baking|cooking|meal prep|coffee brewing)\b", re.I)),
    ("history_education", re.compile(r"\b(scholarships?|colon\w+|empires?|world war \w+|industrial revolution|enlighten\w*)\b", re.I)),
    ("science", re.compile(r"\b(evolution|equilibrium|photosynthesis|black holes?|dna|earthquakes?|ecosystems?|conservation)\b", re.I)),
]


def rule_domain(q):
    for name, rx in DOMAIN_OVERRIDES:
        if name in DOMAINS and rx.search(q):
            return name
    return None


def heuristic_labeler(min_sim, model_name="all-MiniLM-L6-v2"):
    """Free labeler. Less accurate than an LLM, so check it with the audit step."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Missing dependency: pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    descs = [DOMAIN_DESCRIPTIONS.get(d, d.replace("_", " ")) for d in DOMAINS]
    dvec = model.encode(descs, normalize_embeddings=True)

    def label(batch):
        qvec = model.encode([r["query"] for r in batch], normalize_embeddings=True)
        sims = qvec @ dvec.T
        out = {}
        for i, r in enumerate(batch):
            j = int(sims[i].argmax())
            domain = rule_domain(r["query"])
            if domain is None:
                domain = DOMAINS[j] if sims[i][j] >= min_sim else "other"
            out[i] = {"domain": domain, "intent": rule_intent(r["query"]), "flags": rule_flags(r["query"])}
        return out

    return label


def cmd_label(a):
    pool = read_jsonl(a.pool)
    done = set()
    if Path(a.out).exists():
        done = {r["id"] for r in read_jsonl(a.out)}
    todo = [r for r in pool if r["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(done)} already labeled, {len(todo)} to label")
    if a.labeler == "mock":
        labeler = mock_labeler
    elif a.labeler == "heuristic":
        labeler = heuristic_labeler(a.min_sim)
    else:
        labeler = llm_labeler(a.model)

    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        labels = labeler(batch)
        rows = [{**r, **labels[j]} for j, r in enumerate(batch) if j in labels]
        append_jsonl(a.out, rows)
        print(f"  labeled {min(i + a.batch, len(todo))}/{len(todo)}")
    print(f"Done. Labels in {a.out} (re-run the same command to retry any gaps).")


# --------------------------------------------------------------------------- #
# 3. sample
# --------------------------------------------------------------------------- #

def cmd_sample(a):
    rows = read_jsonl(a.labeled)
    flag_counts = Counter()
    eligible = []
    for r in rows:
        for f in r.get("flags", []):
            flag_counts[f] += 1
        if r.get("domain") in DOMAINS and r.get("intent") in INTENTS and not r.get("flags"):
            eligible.append(r)

    cells = defaultdict(list)
    for r in eligible:
        cells[(r["domain"], r["intent"])].append(r)

    rng = random.Random(a.seed)
    need = a.per_cell + a.reserve
    selected, cell_stats, shortfalls = [], {}, []

    for d in DOMAINS:
        for it in INTENTS:
            cand = cells.get((d, it), [])
            by_src = defaultdict(list)
            for r in cand:
                by_src[r["source"]].append(r)
            srcs = sorted(by_src)
            for s in srcs:
                rng.shuffle(by_src[s])
            rng.shuffle(srcs)

            # Round-robin across sources so one dataset cannot dominate a cell.
            order = []
            while any(by_src[s] for s in srcs):
                for s in srcs:
                    if by_src[s]:
                        order.append(by_src[s].pop())

            picked, picked_tokens = [], []
            for r in order:
                t = tokens(r["query"])
                if any(jaccard(t, u) >= NEARDUP_JACCARD for u in picked_tokens):
                    continue
                picked.append(r)
                picked_tokens.append(t)
                if len(picked) == need:
                    break

            for k, r in enumerate(picked):
                selected.append({
                    "query_id": r["id"],
                    "query": r["query"],
                    "domain": d,
                    "intent": it,
                    "source": r["source"],
                    "role": "primary" if k < a.per_cell else "reserve",
                })
            cell_stats[f"{d}|{it}"] = {"eligible": len(cand), "selected": len(picked)}
            if len(picked) < need:
                shortfalls.append({
                    "cell": f"{d}|{it}",
                    "needed_primary": a.per_cell,
                    "got_total": len(picked),
                    "eligible": len(cand),
                    "primary_short": len(picked) < a.per_cell,
                })

    fields = ["query_id", "query", "domain", "intent", "source", "role"]
    write_csv(a.out, selected, fields)

    report = {
        "seed": a.seed,
        "per_cell": a.per_cell,
        "reserve": a.reserve,
        "domains": DOMAINS,
        "intents": INTENTS,
        "labeled_total": len(rows),
        "eligible_total": len(eligible),
        "excluded_by_flag": dict(flag_counts),
        "excluded_unlabeled_or_other": sum(
            1 for r in rows if r.get("domain") not in DOMAINS or r.get("intent") not in INTENTS
        ),
        "selected_total": len(selected),
        "selected_by_source": dict(Counter(r["source"] for r in selected)),
        "cells": cell_stats,
        "shortfalls": shortfalls,
        "output_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
    }
    report_path = Path(a.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    primary_short = [s for s in shortfalls if s["primary_short"]]
    print(f"Selected {len(selected)} queries ({a.per_cell} primary + up to {a.reserve} reserve per cell)")
    print(f"Report: {report_path}")
    if primary_short:
        print(f"WARNING: {len(primary_short)} cells could not fill their primary quota:")
        for s in primary_short:
            print(f"  {s['cell']}: got {s['got_total']}, eligible {s['eligible']}")
        print("Add more candidates for these cells (usually autocomplete/PAA files) and re-run.")


# --------------------------------------------------------------------------- #
# 4. audit
# --------------------------------------------------------------------------- #

def cmd_audit_make(a):
    rows = [r for r in read_csv(a.queries) if r["role"] == "primary"]
    rng = random.Random(a.seed)
    sample = rng.sample(rows, min(a.n, len(rows)))
    out = [{"query_id": r["query_id"], "query": r["query"],
            "human_domain": "", "human_intent": ""} for r in sample]
    write_csv(a.out, out, ["query_id", "query", "human_domain", "human_intent"])
    print(f"Wrote {len(out)} queries to {a.out} (LLM labels hidden to avoid anchoring).")
    print(f"Fill human_domain with one of: {DOMAINS}")
    print(f"Fill human_intent with one of: {INTENTS}")


def cmd_audit_score(a):
    final = {r["query_id"]: r for r in read_csv(a.queries)}
    audit = [r for r in read_csv(a.audit) if r["human_domain"].strip() and r["human_intent"].strip()]
    if not audit:
        sys.exit("No completed rows in the audit file.")

    for field, allowed in (("human_domain", DOMAINS), ("human_intent", INTENTS)):
        bad = [r["query_id"] for r in audit if r[field].strip() not in allowed]
        if bad:
            print(f"WARNING: {len(bad)} rows have a {field} outside the allowed values: {bad[:5]}")

    llm_d = [final[r["query_id"]]["domain"] for r in audit]
    llm_i = [final[r["query_id"]]["intent"] for r in audit]
    hum_d = [r["human_domain"].strip() for r in audit]
    hum_i = [r["human_intent"].strip() for r in audit]

    n = len(audit)
    acc_d = sum(x == y for x, y in zip(llm_d, hum_d)) / n
    acc_i = sum(x == y for x, y in zip(llm_i, hum_i)) / n
    print(f"Rows scored: {n}")
    print(f"Domain: accuracy {acc_d:.1%}, Cohen's kappa {cohen_kappa(llm_d, hum_d):.2f}")
    print(f"Intent: accuracy {acc_i:.1%}, Cohen's kappa {cohen_kappa(llm_i, hum_i):.2f}")
    print("\nDisagreements:")
    for r, ld, li in zip(audit, llm_d, llm_i):
        if ld != r["human_domain"].strip() or li != r["human_intent"].strip():
            print(f"  [{r['query']}]  LLM: {ld}/{li}  human: {r['human_domain']}/{r['human_intent']}")


# --------------------------------------------------------------------------- #
# 5. review (hand-verify the final set)
# --------------------------------------------------------------------------- #

REVIEW_FIELDS = ["query_id", "query", "source", "role", "domain", "intent", "keep", "note"]


def cmd_review_make(a):
    rows = read_csv(a.queries)
    out = [{"query_id": r["query_id"], "query": r["query"], "source": r["source"],
            "role": r["role"], "domain": r["domain"], "intent": r["intent"],
            "keep": "y", "note": ""} for r in rows]
    write_csv(a.out, out, REVIEW_FIELDS)
    print(f"Wrote {len(out)} rows to {a.out}, prefilled with the script's labels.")
    print("Fix wrong domain/intent values, and set keep to n for unusable queries")
    print("(malformed, ambiguous, off-topic). Leave correct rows alone.")
    print(f"domain: {DOMAINS}")
    print(f"intent: {INTENTS}")


def cmd_review_apply(a):
    orig = {r["query_id"]: r for r in read_csv(a.queries)}
    order = {qid: i for i, qid in enumerate(orig)}
    kept, dropped, invalid = [], [], []
    changed_domain = changed_intent = 0

    for r in read_csv(a.review):
        qid = r["query_id"]
        if qid not in orig:
            invalid.append(qid)
            continue
        if r["keep"].strip().lower() in ("n", "no", "0", "drop"):
            dropped.append(qid)
            continue
        d, it = r["domain"].strip(), r["intent"].strip()
        if d not in DOMAINS or it not in INTENTS:
            invalid.append(qid)
            continue
        o = orig[qid]
        changed_domain += d != o["domain"]
        changed_intent += it != o["intent"]
        kept.append({"query_id": qid, "query": o["query"], "source": o["source"],
                     "domain": d, "intent": it,
                     "_rank": (o["role"] != "primary", order[qid])})

    cells = defaultdict(list)
    for r in kept:
        cells[(r["domain"], r["intent"])].append(r)

    final, short = [], []
    for d in DOMAINS:
        for it in INTENTS:
            rows = sorted(cells.get((d, it), []), key=lambda r: r["_rank"])
            for k, r in enumerate(rows):
                final.append({"query_id": r["query_id"], "query": r["query"], "domain": d,
                              "intent": it, "source": r["source"],
                              "role": "primary" if k < a.per_cell else "reserve"})
            if len(rows) < a.per_cell:
                short.append({"cell": f"{d}|{it}", "primary_filled": len(rows), "needed": a.per_cell})

    write_csv(a.out, final, ["query_id", "query", "domain", "intent", "source", "role"])
    n_reviewed = len(kept) + len(dropped)
    report = {
        "rows_reviewed": n_reviewed,
        "kept": len(kept),
        "dropped": len(dropped),
        "invalid_rows_excluded": invalid,
        "domain_corrections": changed_domain,
        "intent_corrections": changed_intent,
        "domain_label_accuracy_before_review": round(1 - changed_domain / max(1, len(kept)), 3),
        "intent_label_accuracy_before_review": round(1 - changed_intent / max(1, len(kept)), 3),
        "primary_total": sum(r["role"] == "primary" for r in final),
        "cells_short_of_quota": short,
        "output_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
    }
    report_path = Path(a.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Kept {len(kept)}, dropped {len(dropped)}; corrections: {changed_domain} domain, {changed_intent} intent")
    print(f"Primary queries: {report['primary_total']}; cells short of {a.per_cell}: {len(short)}")
    if invalid:
        print(f"WARNING: {len(invalid)} rows had invalid values and were excluded: {invalid[:5]}")
    print(f"Wrote {a.out} and {report_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("pool", help="build the candidate query pool")
    s.add_argument("--nq", type=int, default=4000, help="raw rows to pull from Natural Questions")
    s.add_argument("--msmarco", type=int, default=4000, help="raw rows to pull from MS MARCO")
    s.add_argument("--extra", nargs="*", help="CSV files with a 'query' column (autocomplete, PAA)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="pool.jsonl")
    s.set_defaults(fn=cmd_pool)

    s = sub.add_parser("label", help="label domain, intent and flags")
    s.add_argument("--pool", default="pool.jsonl")
    s.add_argument("--out", default="labeled.jsonl")
    s.add_argument("--model", default=os.environ.get("LABEL_MODEL", DEFAULT_MODEL))
    s.add_argument("--batch", type=int, default=25)
    s.add_argument("--limit", type=int, default=0, help="label at most this many (0 = all)")
    s.add_argument("--labeler", choices=["llm", "heuristic", "mock"], default="llm",
                   help="llm needs an API key; heuristic is free; mock is for pipeline testing only")
    s.add_argument("--min-sim", type=float, default=0.2,
                   help="heuristic labeler: minimum similarity for a domain match, else 'other'")
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("sample", help="sample the domain x intent grid")
    s.add_argument("--labeled", default="labeled.jsonl")
    s.add_argument("--per-cell", type=int, default=4)
    s.add_argument("--reserve", type=int, default=2)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="queries_final.csv")
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("audit-make", help="create a blind hand-labeling sheet")
    s.add_argument("--queries", default="queries_final.csv")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="audit_blind.csv")
    s.set_defaults(fn=cmd_audit_make)

    s = sub.add_parser("audit-score", help="score LLM labels against your hand labels")
    s.add_argument("--audit", default="audit_blind.csv")
    s.add_argument("--queries", default="queries_final.csv")
    s.set_defaults(fn=cmd_audit_score)

    s = sub.add_parser("review-make", help="create a prefilled sheet to hand-verify the final set")
    s.add_argument("--queries", default="queries_final.csv")
    s.add_argument("--out", default="review.csv")
    s.set_defaults(fn=cmd_review_make)

    s = sub.add_parser("review-apply", help="apply your corrections and rebuild the final set")
    s.add_argument("--queries", default="queries_final.csv")
    s.add_argument("--review", default="review.csv")
    s.add_argument("--per-cell", type=int, default=4)
    s.add_argument("--out", default="queries_reviewed.csv")
    s.set_defaults(fn=cmd_review_apply)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()