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


# Every echo-shaped study, newest last. A study missing from this list is simply
# absent from the report -- which is how the 10k rerun and the full-attention
# baseline stayed invisible for a day after they finished.
ECHO_STUDIES = (
    ("vv_*.json", "V;V (T=64)"),
    ("base_*.json", "denominateur (T=64)"),
    ("cap_*.json", "capacite (T=64)"),
    ("L_*.json", "flux longs (T=256)"),
    ("K_*.json", "convergence 10000 pas (T=64)"),
    ("F_*.json", "attention pleine seg>=2T"),
)


def section_echo(root, out_dir, lines):
    """mem_effect (headline) and memory_gain per arm, over every echo-shaped sweep.

    ``mem_effect = L1(memory off) - L1(repeat)`` is the half of ``memory_gain`` that
    is authentic recall; ``memory_gain`` also counts the noise the memory injects in
    the mismatched condition, which is a third of it. Both are reported, mem_effect
    first, because the comparative ordering is the same but the amplitude is not.
    """
    rows = []
    for pat, study in ECHO_STUDIES:
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
        me, me_sem, n = agg([x.get("mem_effect") for x in v])
        mg, sem, _ = agg([x["memory_gain"] for x in v])
        c = v[0].get("cost") or {}
        out.append([study, label, n,
                    round(me, 4) if me is not None else "",
                    round(me_sem, 4) if me_sem is not None else "",
                    round(mg, 4), round(sem, 4),
                    v[0].get("steps") or (v[0].get("cost") or {}).get("steps") or "",
                    c.get("segment_size"), v[0].get("use_memory"),
                    c.get("state_bytes"), c.get("kv_bytes"), c.get("kv_bytes_full"),
                    round(c.get("s_per_step", 0), 2), round(c.get("peak_vram_gib", 0), 2)])
    p = write_csv(os.path.join(out_dir, "memory_gain_et_cout.csv"),
                  ["etude", "bras", "n", "mem_effect", "sem_mem_effect",
                   "memory_gain", "sem", "pas", "seg", "memoire",
                   "etat_octets", "kv_octets", "kv_plein_octets", "s_par_pas",
                   "vram_gio"], out)
    lines.append(f"- `{os.path.basename(p)}` : mem_effect (métrique de tête) et "
                 f"memory_gain par bras, avec le coût ({len(out)} bras).")


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


GAPS = (0, 64, 128, 256, 512, 1024, 2048)


def section_retention(root, out_dir, lines):
    """The retention curve read in ``mem_effect``, with the survivor count kept.

    ``retention_gain`` (= memory_gain at a distance) stayed positive out to 2048
    latents, but it is a *difference of two conditions that are both degraded*. The
    curve that answers "is carrying a memory across this stream worth it at all" is
    ``mem_effect(n) = l1_repeat_memoff - l1_repeat``, and it is already stored in
    every retention file -- switching metric costs no GPU.

    Seeds that went non-finite are dropped from the mean and counted separately
    rather than folded in, so a column that looks good on one surviving seed cannot
    be read as a column that looks good.
    """
    rows = load_dicts(os.path.join(root, "retention_all.json"))
    if not rows:
        for p in glob.glob(os.path.join(root, "retention_out", "*.json")):
            rows += load_dicts(p)
    per = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in rows:
        if "curve" not in r:
            continue
        # The same artifact is scored by more than one job (the chain's per-arm
        # pass and the later per-seed one both cover seed 0), and counting it
        # twice inflates the survivor denominator the whole table is read against.
        key = (r.get("file"), r.get("T"), r.get("horizon"))
        if key in seen:
            continue
        seen.add(key)
        # Keyed on (arm, T): the long-stream study reuses the arm names of the
        # T=64 sweep, so keying on the name alone silently averaged a 256-latent
        # model into a 64-latent one and inflated the survivor counts.
        arm = f"{r.get('arm', '?')}@T{r.get('T', '?')}"
        for c in r["curve"]:
            per[arm][c["gap_latents"]].append(
                (c["l1_repeat_memoff"] - c["l1_repeat"], c["retention_gain"],
                 c.get("status")))
    if not per:
        return
    out = []
    for arm, by_gap in sorted(per.items()):
        row = [arm]
        for g in GAPS:
            v = by_gap.get(g)
            if not v:
                row += ["", "", "", ""]
                continue
            me, _, n_ok = agg([x[0] for x in v])
            rg, _, _ = agg([x[1] for x in v])
            # "sain" is only available on curves scored after the health probe was
            # added; older files report finiteness only, and say so by leaving the
            # column blank rather than by claiming every finite seed was healthy.
            st = [x[2] for x in v if x[2]]
            n_sain = f"{sum(s == 'sain' for s in st)}/{len(st)}" if st else ""
            row += [round(me, 4) if me is not None else "NaN",
                    round(rg, 4) if rg is not None else "NaN",
                    f"{n_ok}/{len(v)}", n_sain]
        out.append(row)
    header = ["bras"] + [f"{k}_+{g}" for g in GAPS
                         for k in ("mem_effect", "retention_gain", "graines_finies",
                                   "graines_saines")]
    p = write_csv(os.path.join(out_dir, "retention_mem_effect.csv"), header, out)
    lines.append(f"- `{os.path.basename(p)}` : rétention en mem_effect (métrique de "
                 f"tête) et en retention_gain, avec le nombre de graines finies "
                 f"par écart ({len(out)} bras).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="folder holding the pulled JSONs")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    lines = []
    for fn in (section_echo, section_retention, section_margins, section_length,
               section_converge):
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
