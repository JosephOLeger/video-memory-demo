"""Volet R2b : reentrainement lucidrains (train_r_prlaw a l'identique,
graine 0, 4000 pas) avec la regle de magnitude usq active pendant TOUT
l'entrainement. Checkpoints renommes ckpt_psic_* par train_r_psic_body.py
(sed de train_r_prlaw.py fait au deploiement — seule difference).

Prediction P-R2b : c'est le vrai test de transfert du levier — chez nous
la regle agit en changeant ce que l'entrainement fait des statistiques
d'ecriture (r_eff stable le long des checkpoints), pas le rejeu.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_psic

patch_psic.install()
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "train_r_psic_body.py"),
    run_name="__main__",
)
