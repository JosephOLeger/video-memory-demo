"""Tableaux de bilan_long : precision (%) +- erreur standard binomiale, par condition et par bin."""
import json
import math
import sys
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "res"
ARMS = ["deep_s0", "deep_s1", "deep_s2", "pool_s0", "poolP_s0", "fifo_s0", "mean_s0"]
CONDS = ["orig", "L8", "L32", "L80", "D2", "D3"]


def load(cond):
    try:
        return [json.loads(l) for l in open(f"{RES}/{cond}.jsonl")]
    except FileNotFoundError:
        return []


def pct(rows, f):
    n = len(rows)
    if not n:
        return "-"
    p = sum(f(r) for r in rows) / n
    return f"{100 * p:.0f} ± {100 * math.sqrt(p * (1 - p) / n):.0f}"


def table(title, groups, arms):
    print(f"\n### {title}\n")
    print("| groupe | n | " + " | ".join(arms) + " | fenêtre | enseignant |")
    print("|---|---|" + "---|" * (len(arms) + 2))
    for name, rows in groups:
        if not rows:
            continue
        cells = [pct(rows, lambda r, a=a: r[a] == r["gold"]) + " / " +
                 pct(rows, lambda r, a=a: r[a + "_wrong"] == r["gold"]).split(" ")[0] for a in arms if a in rows[0]]
        print(f"| {name} | {len(rows)} | " + " | ".join(cells) + f" | {pct(rows, lambda r: r['window'] == r['gold'])}"
              f" | {pct(rows, lambda r: r['teacher'] == r['gold'])} |")


def lure_rate(rows, arm):
    """part des reponses qui tombent sur une couleur leurre."""
    n = len(rows)
    p = sum(r["options"][r[arm]] in r["lures"] for r in rows) / n
    return f"{100 * p:.0f}"


def main():
    for cond in CONDS:
        rows = load(cond)
        if not rows:
            continue
        arms = [a for a in ARMS if a in rows[0]]
        groups = [("tout", rows)]
        bins = sorted({r["bin"] for r in rows}, key=lambda b: int(b.strip("d").split("-")[0]))
        if cond == "orig":
            bins_ = [("dist 2-3", [r for r in rows if r["dist"][0] < 4]),
                     ("dist 4-8", [r for r in rows if r["dist"][0] >= 4])]
            groups += bins_
            groups.append(("dur (fenêtre fausse)", [r for r in rows if r["window"] != r["gold"]]))
        elif cond.startswith("L"):
            groups += [(f"dist {b}", [r for r in rows if r["bin"] == b]) for b in bins]
            groups.append(("dur (fenêtre fausse)", [r for r in rows if r["window"] != r["gold"]]))
        else:
            for w in ("first", "last"):
                groups.append((f"« {w} »", [r for r in rows if r["which"] == w]))
            groups.append(("enseignant juste", [r for r in rows if r["teacher"] == r["gold"]]))
        table(f"{cond} (mémoire / autre vidéo, en %)", groups, arms)
        if cond.startswith("D"):
            print("\nPart des réponses sur une couleur leurre (%) : " + ", ".join(
                f"{a} {lure_rate(rows, a)}" for a in arms + ["window", "teacher"]))
            for w in ("first", "last"):
                sub = [r for r in rows if r["which"] == w]
                if sub:
                    print(f"  « {w} » : " + ", ".join(f"{a} {lure_rate(sub, a)}" for a in arms + ["teacher"]))
        secs = [r.get("sec", 0) for r in rows]
        print(f"\n(n écritures moyen {sum(r['n_writes'] for r in rows) / len(rows):.0f} ; "
              f"{sum(secs) / len(secs):.1f} s/question)")


if __name__ == "__main__":
    main()
