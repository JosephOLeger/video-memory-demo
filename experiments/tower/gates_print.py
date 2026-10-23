import json,glob
for f in sorted(glob.glob("*.json")):
    if "smoke" in f or "queue" in f: continue
    try: d=json.load(open(f))
    except Exception: continue
    g=d.get("gates") if isinstance(d,dict) else None
    if not g: continue
    x=g[-1]; e,th,a=x["eta_med"],x["theta_med"],x["alpha_med"]
    print(f"{f[:-5]:18s} step {x['step']:5d} eta {e:.3f} alpha {a:.1e} theta {th:.2e} th/(1-eta) {th/max(1-e,1e-9):.2e} pr {x['pr_w1']:.1f}")
