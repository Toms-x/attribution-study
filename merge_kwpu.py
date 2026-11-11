import csv, glob, os

SKIP_PREFIX = "keywordpeopleuse_seed"
BAD_SEEDS = {"keyword"}
rows, seen = [], set()


def add(q, src):
    q = " ".join((q or "").split())
    if q and q.lower() not in seen:
        seen.add(q.lower())
        rows.append({"query": q, "source": src})
        return 1
    return 0


for f in sorted(glob.glob("kwpu/*.csv")):
    name = os.path.basename(f)
    if name.startswith(SKIP_PREFIX):
        continue
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        seed_col = cols.get("query")
        n = skipped = 0
        if "suggestion" in cols:
            src = "google_autocomplete"
            for r in rd:
                if seed_col and (r.get(seed_col) or "").strip().lower() in BAD_SEEDS:
                    skipped += 1
                    continue
                n += add(r.get(cols["suggestion"]), src)
        else:
            src = "google_paa"
            qcols = [c for k, c in cols.items() if k.startswith(("question", "qustion"))]
            for r in rd:
                if seed_col and (r.get(seed_col) or "").strip().lower() in BAD_SEEDS:
                    skipped += 1
                    continue
                for c in qcols:
                    n += add(r.get(c), src)
    print(f"{name}: {n} new queries ({src}), {skipped} junk rows skipped")

with open("autocomplete.csv", "w", newline="", encoding="utf-8") as o:
    w = csv.DictWriter(o, fieldnames=["query", "source"])
    w.writeheader()
    w.writerows(rows)
print("autocomplete.csv:", len(rows), "rows")
