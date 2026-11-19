"""Volet R2a : la sonde volet-R avec la regle de magnitude usq active a
l'INFERENCE seulement (ckpt entraine sans la regle).

Prediction P-R2a : ne repare PAS le collapse intra-sequence (la
correlation directionnelle des ecritures est apprise ; une regle
scalaire au rejeu ne decorrele rien). Si FAUX, le levier est mecanique
et trivialement portable — a consigner tel quel.

Usage : memes arguments que probe_r_prlaw.py.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_psic

patch_psic.install()
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "probe_r_prlaw.py"),
    run_name="__main__",
)
