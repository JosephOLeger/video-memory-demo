"""La cellule manquante du 2x2 direction x magnitude (post-purge).

Cellules connues sur bases saines : NS + magnitude bornee (usq, tient) et
brut + brut (Titans-litteral, meurt 8/8). La cellule qui prouverait que la
magnitude SUFFIT — direction brute + magnitude bornee — n'existe pas :
l'ancienne preuve inverse passait par l'hybride rescale, purge.

Meme approche script-only que diag_decouple_train.py : on patche
NeuralMemory._unit_update pour remplacer la direction NS-5(S) par S
normalise en Frobenius, en gardant la regle de magnitude usq
psi_c(||S||) et le meme c (--muon unit_squash --muon-scale 350). La borne
spectrale ||dM||_2 <= ||dM||_F = theta_t * psi < theta_t tient toujours
(Frobenius domine le spectral) ; sur des ecritures quasi rang-1
(r_eff 1.6-3.6 mesures) les deux normes coincident, donc le depot est
apparie a usq. On evite exprimer la norme spectrale exacte (svdvals dans
la boucle interne differentiable = instabilite du backward que NS-5
existe precisement pour eviter).

Prediction pre-enregistree P-2x2 (predictions_collapse_2026-08-28.md).

Usage : memes arguments que titans.worldmodel.echo, config famille 10k :
  diag_nons_train.py --label rawdir_usq_s0 --train ... --val ...
    --horizon 24 --steps 10000 --seed 0 --eta-init 0.995
    --theta-scale 0.03 --alpha-init 0.002 --muon unit_squash
    --muon-scale 350 --frozen-every 2000 --out ... --frozen-out ...
"""
import math

import torch

import titans.memory as tmem
import titans.worldmodel.echo as echo


def _rawdir_unit_update(self, raw: torch.Tensor) -> torch.Tensor:
    if raw.dim() >= 3:
        u = raw / (raw.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        n = raw.norm(dim=(-2, -1), keepdim=True)
    else:
        # Les biais traversent NS-5 inchanges dans usq (memory.py:900) ;
        # on les laisse inchanges ici aussi pour rester apparie.
        u = raw
        n = raw.norm(dim=-1, keepdim=True)
    if self._orth_mode != "unit_squash":
        return u
    c = self.orthogonalize_clip
    return u * (2.0 / math.pi) * torch.atan(n * (math.pi / (2.0 * c)))


tmem.NeuralMemory._unit_update = _rawdir_unit_update

if __name__ == "__main__":
    echo.main()
