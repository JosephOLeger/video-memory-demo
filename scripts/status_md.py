"""One page saying what ran overnight and what it found, for a reader with no shell.

The results folder already carries the CSVs and the figures. What it does not carry
is the answer to "how far along is this, and did anything break" -- which is the
first question on coming back to a machine that ran unattended. This writes that
page from the mirrored JSONs alone, so it is honest about what has actually landed
rather than about what was queued.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

TOK = 64


def _load(pattern):
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        yield p, d


def tau_of(eta):
    return None if eta is None else 1.0 / (1.0 - eta)


def section_tau(root, out):
    """The critical accumulation horizon, per seed -- the point of the night's sweep."""
    probes = defaultdict(list)
    for _, d in _load(os.path.join(root, "taucrit", "*.json")):
        sm = (d.get("summary") or {}).get("loop")
        if sm and not d.get("cascade"):
            probes[d.get("eta_init")].append(not sm["all_finite"])
    for _, d in _load(os.path.join(root, "vvlong_nocas_e*.json")):
        sm = (d.get("summary") or {}).get("loop")
        if sm:
            probes.setdefault(d.get("eta_init"), [])
            probes[d.get("eta_init")].append(not sm["all_finite"])
    if not probes:
        out.append("_(aucune sonde de survie n'a encore atterri)_\n")
        return
    out.append("| η | τ_acc (tokens) | graines | divergent | fraction |")
    out.append("|---|---|---|---|---|")
    lo = hi = None
    for eta in sorted(k for k in probes if k is not None):
        v = probes[eta]
        frac = sum(v) / len(v)
        out.append("| %.3f | %.0f | %d | %d | **%.2f** |"
                   % (eta, tau_of(eta), len(v), sum(v), frac))
        if frac < 0.5:
            lo = tau_of(eta)
        elif hi is None:
            hi = tau_of(eta)
    out.append("")
    if lo and hi:
        out.append("**τ\\* est encadré entre %.0f et %.0f tokens.** "
                   "(τ_acc = 1/(1−η) ; la fraction est sur les graines, "
                   "car la divergence est dépendante de la graine.)\n" % (lo, hi))
    else:
        out.append("_La transition n'est pas encore encadrée par ces points._\n")


def section_runs(root, out):
    """What landed, per study, from the result files themselves."""
    groups = [("E_e*.json", "balayage τ* (η intermédiaires)"),
              ("A_T*.json", "sweep longueur d'entraînement"),
              ("NH_*.json", "trajectoires normalize_hidden"),
              ("B_cv40k*.json", "trajectoire 40 000 pas")]
    out.append("| étude | cellules | mem_effect moyen |")
    out.append("|---|---|---|")
    for pat, name in groups:
        rs = [d for _, d in _load(os.path.join(root, pat))
              if isinstance(d.get("mem_effect"), (int, float))]
        me = ("%.4f" % st.mean([r["mem_effect"] for r in rs])) if rs else "—"
        out.append("| %s | %d | %s |" % (name, len(rs), me))
    out.append("")


def section_theta(root, out):
    """The gates the outer loop chose -- the cascade's instability lives here.

    theta and alpha are freely learned, so what they settle at is a *measurement* of
    what the architecture forced, not a hyperparameter. The cascade arms land ~18x
    higher in theta and ~8x lower in alpha than the scalar ones, which is what carries
    them past the stability bound while the leak rate does nothing.
    """
    p = os.path.join(root, "theta_audit.json")
    try:
        recs = [r for r in json.load(open(p)) if "error" not in r]
    except Exception:
        out.append("_(pas encore d'audit theta/alpha)_\n")
        return
    by = defaultdict(list)
    for r in recs:
        by[r.get("label") or "?"].append(r)
    rows = []
    for lab, v in by.items():
        def med(k):
            xs = sorted(x[k] for x in v if isinstance(x.get(k), (int, float)))
            return xs[len(xs) // 2] if xs else float("nan")
        a = med("alpha_t_mean")
        rows.append((med("theta_t_max"), lab, len(v), a,
                     (0.6931 / a) if a else float("inf"), med("margin"),
                     bool(v[0].get("cascade"))))
    rows.sort(reverse=True)
    out.append("| bras | n | θ_max | α_moyen | τ_oubli | marge |")
    out.append("|---|---|---|---|---|---|")
    for th, lab, n, a, tau, mar, cas in rows:
        out.append("| %s%s | %d | %.4f | %.6f | %.0f | %.2f |"
                   % ("**" if cas else "", lab + ("**" if cas else ""),
                      n, th, a, tau, mar))
    out.append("")
    out.append("_En gras : les bras cascade. θ et α sont librement appris — ce tableau "
               "dit ce que l'architecture a **forcé** la boucle externe à choisir._\n")


def section_broken(root, out):
    """Anything that errored, named. A silent failure is the expensive kind."""
    bad = []
    for p, d in _load(os.path.join(root, "retention_out", "*.json")):
        for r in (d if isinstance(d, list) else [d]):
            if isinstance(r, dict) and "error" in r:
                bad.append("`%s` — %s" % (r.get("file", os.path.basename(p)),
                                          str(r["error"])[:110]))
    if bad:
        out.append("### Cellules en erreur\n")
        out += ["- " + b for b in bad[:12]] + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = ["# État — régénéré automatiquement", "",
           "Cette page est écrite à partir des fichiers de résultats réellement "
           "présents, pas de ce qui a été mis en file.", "",
           "## Le seuil τ\\* — l'objet du balayage de la nuit", ""]
    section_tau(a.root, out)
    out += ["## Les portes que la boucle externe a choisies", ""]
    section_theta(a.root, out)
    out += ["## Ce qui a atterri", ""]
    section_runs(a.root, out)
    section_broken(a.root, out)
    out += ["## Les fichiers", "",
            "- `compromis_reactivite_stabilite.png` — la figure de tête : réactivité, "
            "seuil de divergence, et les deux signatures (norme qui explose / rang "
            "qui tombe).",
            "- `retention_curve.png` — rétention épisodique en fonction de la distance.",
            "- `retention_mem_effect.csv` — la même chose en `mem_effect`, avec le "
            "nombre de graines finies par écart.",
            "- `memory_gain_et_cout.csv` — toutes les études d'écho, avec le coût.",
            "- `convergence.csv`, `marge_stabilite.csv`, `survie_longueur.csv`.", ""]
    with open(a.out, "w") as fh:
        fh.write("\n".join(out))
    print(a.out)


if __name__ == "__main__":
    main()
