"""Collect every finished study into one readable folder: figures, CSVs, a summary.

Results live on the workstation as a few hundred JSON files whose names encode the
arm. Nobody can read that, and the person who needs to read it does not have a shell
on the box. So this renders what exists -- skipping silently what does not -- into a
directory that can sit on a laptop.

Deliberately tolerant: a study that has not finished, or a file that is malformed,
drops out of the report instead of aborting it. The point is to be runnable at any
moment during a multi-day queue and produce whatever is true so far.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict


def load_dicts(pattern):
    out = []
    for p in glob.glob(pattern):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, list):
            out.extend(x for x in r if isinstance(x, dict))
    return out


def agg(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None, None, 0
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return st.mean(vals), sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0, len(vals)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def section_echo(root, out_dir, lines):
    """memory_gain and cost per arm, over every sweep that used the echo cells."""
    rows = []
    for pat, study in (("vv_*.json", "V;V (T=64)"), ("base_*.json", "denominateur (T=64)"),
                       ("cap_*.json", "capacite (T=64)"), ("L_*.json", "flux longs (T=256)")):
        for r in load_dicts(os.path.join(root, pat)):
            if not isinstance(r.get("memory_gain"), (int, float)):
                continue
            rows.append((study, r))
    if not rows:
        return
    g = defaultdict(list)
    for study, r in rows:
        g[(study, r.get("label"))].append(r)

    out = []
    for (study, label), v in sorted(g.items()):
        mg, sem, n = agg([x["memory_gain"] for x in v])
        c = v[0].get("cost") or {}
        out.append([study, label, n, round(mg, 4), round(sem, 4),
                    c.get("segment_size"), v[0].get("use_memory"),
                    c.get("state_bytes"), c.get("kv_bytes"), c.get("kv_bytes_full"),
                    round(c.get("s_per_step", 0), 2), round(c.get("peak_vram_gib", 0), 2)])
    p = write_csv(os.path.join(out_dir, "memory_gain_et_cout.csv"),
                  ["etude", "bras", "n", "memory_gain", "sem", "seg", "memoire",
                   "etat_octets", "kv_octets", "kv_plein_octets", "s_par_pas",
                   "vram_gio"], out)
    lines.append(f"- `{os.path.basename(p)}` : memory_gain et coût par bras "
                 f"({len(out)} bras).")


def section_margins(root, out_dir, lines):
    ms = load_dicts(os.path.join(root, "vv_margins.json"))
    ms = [m for m in ms if m.get("all_finite")]
    if not ms:
        return
    g = defaultdict(list)
    for m in ms:
        g[m.get("label")].append(m)
    out = []
    for label, v in sorted(g.items()):
        mm, _, n = agg([x["margin"] for x in v])
        pr, _, _ = agg([x["pr"] for x in v])
        cr, _, _ = agg([x["cos_ret"] for x in v])
        tt, _, _ = agg([x["theta_t_max"] for x in v])
        out.append([label, n, round(mm, 2), round(pr, 1), round(cr, 3), round(tt, 5)])
    p = write_csv(os.path.join(out_dir, "marge_stabilite.csv"),
                  ["bras", "n", "marge", "pr", "cos_ret", "theta_t_max"], out)
    lines.append(f"- `{os.path.basename(p)}` : marge de stabilité "
                 f"(2.37 / max θ·λ ; < 1 = au-delà du seuil).")


def section_length(root, out_dir, lines):
    rows = []
    for pat, reps, T in (("vvlong_*.json", 1000, 64), ("long5k_*.json", 5000, 144)):
        for d in load_dicts(os.path.join(root, pat)):
            for mode in ("loop", "cycle", "shuffle"):
                tr = (d.get("streams") or {}).get(mode)
                if not tr:
                    continue
                fin = [x for x in tr if x["finite"]]
                s = (d.get("summary") or {}).get(mode, {})
                rows.append([f"{reps}x{T}", d.get("label"), mode,
                             "" if len(fin) >= reps else len(fin),
                             len(fin) * T, round(s.get("l1_tail", float("nan")), 4),
                             round(s.get("memory_gain_tail", float("nan")), 4)])
    if not rows:
        return
    p = write_csv(os.path.join(out_dir, "survie_longueur.csv"),
                  ["sonde", "bras", "flux", "rupture_rep", "tokens_tenus",
                   "l1_fin", "gain_fin"], rows)
    lines.append(f"- `{os.path.basename(p)}` : survie en longueur "
                 f"(rupture vide = a tenu jusqu'au bout).")


def section_converge(root, out_dir, lines):
    rows = load_dicts(os.path.join(root, "cv_converge.json"))
    if not rows:
        return
    out = [[r.get("arm"), r.get("step"), round(r.get("memory_gain", float("nan")), 4),
            round(r.get("margin", float("nan")), 2),
            round(r.get("theta_t_max", float("nan")), 5),
            round(r.get("second", float("nan")), 4)]
           for r in rows if "error" not in r]
    if not out:
        return
    p = write_csv(os.path.join(out_dir, "convergence.csv"),
                  ["bras", "pas", "memory_gain", "marge", "theta_t_max", "l1"], out)
    lines.append(f"- `{os.path.basename(p)}` : les deux axes le long de "
                 f"l'entraînement — dit si 2500 pas est un équilibre.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="folder holding the pulled JSONs")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    lines = []
    for fn in (section_echo, section_margins, section_length, section_converge):
        try:
            fn(a.root, a.out_dir, lines)
        except Exception as e:                     # one broken study must not
            lines.append(f"- (échec {fn.__name__}: {repr(e)[:80]})")   # kill the rest

    ret = os.path.join(a.root, "retention_all.json")
    if os.path.exists(ret):
        lines.append("- `retention_curve.png` / `.csv` : rétention épisodique "
                     "en fonction de la distance.")

    readme = os.path.join(a.out_dir, "LISEZ-MOI.md")
    with open(readme, "w") as fh:
        fh.write("# Résultats\n\nGénéré automatiquement. Ce qui est présent :\n\n")
        fh.write("\n".join(lines) if lines else "- (rien de terminé pour l'instant)")
        fh.write("\n")
    print(readme)
    for l in lines:
        print(" ", l)


if __name__ == "__main__":
    main()
