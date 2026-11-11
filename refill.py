import csv, json, sys

REVIEW, FINAL, LABELED = "review.csv", "queries_final.csv", "labeled.jsonl"
RF = ["query_id", "query", "source", "role", "domain", "intent", "keep", "note"]
FF = ["query_id", "query", "domain", "intent", "source", "role"]
DOMAINS = {"health", "personal_finance", "consumer_tech", "science", "law_government",
           "travel", "food_cooking", "history_education", "home_diy", "consumer_products"}
INTENTS = {"factual_lookup", "how_to", "comparison", "explanation", "recommendation"}


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def labeled():
    out = {}
    with open(LABELED, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


cmd = sys.argv[1] if len(sys.argv) > 1 else ""

if cmd == "find":
    text = sys.argv[2].lower()
    for r in read(REVIEW):
        if text in r["query"].lower():
            print(f'{r["query_id"]} | keep={r["keep"]} | {r["domain"]}/{r["intent"]} | {r["query"]}')

elif cmd == "list":
    domain, intent = sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    used = {r["query_id"] for r in read(FINAL)}
    n = 0
    for r in labeled().values():
        if (r["domain"], r["intent"]) == (domain, intent) and not r["flags"] and r["id"] not in used:
            n += 1
            if n <= limit:
                print(r["id"], "|", r["source"], "|", r["query"])
    print("candidates:", n)

elif cmd == "add":
    lab, final, review = labeled(), read(FINAL), read(REVIEW)
    in_final = {r["query_id"] for r in final}
    in_review = {r["query_id"] for r in review}
    for arg in sys.argv[2:]:
        qid, _, override = arg.partition("=")
        if qid not in lab:
            print("unknown id:", qid)
            continue
        r = lab[qid]
        domain, intent, note = r["domain"], r["intent"], "added from pool"
        if override:
            domain, _, intent = override.partition("/")
            if domain not in DOMAINS or intent not in INTENTS:
                print("bad label for", qid, "->", override)
                continue
            note = "added from pool, labeled by hand"
        if qid in in_final and qid in in_review and not override:
            print("already present:", qid)
            continue
        final = [x for x in final if x["query_id"] != qid]
        review = [x for x in review if x["query_id"] != qid]
        final.append({"query_id": qid, "query": r["query"], "domain": domain,
                      "intent": intent, "source": r["source"], "role": "reserve"})
        review.append({"query_id": qid, "query": r["query"], "source": r["source"],
                       "role": "reserve", "domain": domain, "intent": intent,
                       "keep": "y", "note": note})
        in_final.add(qid)
        in_review.add(qid)
        print("added:", qid, "|", domain, "/", intent, "|", r["query"])
    write(FINAL, final, FF)
    write(REVIEW, review, RF)

elif cmd == "drop":
    review = read(REVIEW)
    ids = set(sys.argv[2:])
    hit = 0
    for r in review:
        if r["query_id"] in ids:
            r["keep"] = "n"
            hit += 1
    write(REVIEW, review, RF)
    print(f"marked keep=n on {hit} of {len(ids)} ids")

elif cmd == "relabel":
    review = read(REVIEW)
    changes = {}
    for arg in sys.argv[2:]:
        qid, _, ov = arg.partition("=")
        d, _, i = ov.partition("/")
        if (d and d not in DOMAINS) or (i and i not in INTENTS):
            print("bad label for", qid, "->", ov)
            continue
        changes[qid] = (d, i)
    done = set()
    for r in review:
        if r["query_id"] in changes:
            d, i = changes[r["query_id"]]
            r["domain"] = d or r["domain"]
            r["intent"] = i or r["intent"]
            done.add(r["query_id"])
            print("relabeled:", r["query_id"], "->", r["domain"], "/", r["intent"], "|", r["query"])
    write(REVIEW, review, RF)
    for qid in changes:
        if qid not in done:
            print("not in review.csv:", qid)

else:
    print("usage: refill.py find <text> | list <domain> <intent> [max] | add <id[=domain/intent]> ... | drop <ids>")
