"""Bras causal du collapse : mu_unit10k + decouple_memory_update.

Meme approche script-only que diag_titans_train.py (aucune modification
de src) : on patche echo.WorldModel pour injecter
``decouple_memory_update=True`` (mac.py:106 — coupe la boucle
lecture->attention->ecriture identifiee comme fabriquant la correlation
de cles rho_k et le verrouillage rang-1), puis on delegue a echo.main().

Prediction pre-enregistree (P-C1/P-C2,
results/predictions_collapse_2026-08-28.md) : pr plus haut a config
egale, via rho_k plus bas et/ou g plus haut ; memory_gain dans le bruit
de graine.

Usage : memes arguments que titans.worldmodel.echo, config mu_unit10k :
  diag_decouple_train.py --label mu_unit10k_dec_s0 --train ... --val ...
    --horizon 24 --steps 10000 --seed 0 --eta-init 0.995
    --theta-scale 0.03 --alpha-init 0.002 --muon unit
    --frozen-every 2000 --out ... --frozen-out ...
"""
import titans.worldmodel.echo as echo

_Orig = echo.WorldModel


class _DecoupledWorldModel(_Orig):
    def __init__(self, *args, **kwargs):
        kwargs["decouple_memory_update"] = True
        super().__init__(*args, **kwargs)


echo.WorldModel = _DecoupledWorldModel

if __name__ == "__main__":
    echo.main()
