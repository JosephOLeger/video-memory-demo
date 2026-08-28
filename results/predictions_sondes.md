# Predictions — sondes A (lecture) et B (trace mecanisme), 2026-08-26

Enregistrees AVANT lancement (protocole standard). Contexte : verdicts du
26/08 dans predictions_d2interne.md — P-F2a tombee (etat fini partout sur le
protocole echo, scores NaN aux gaps >= 4 blocs avec pr qui fond), et
echelle 2 close (theta* dans (0.05, 0.08], morts toutes D1 au forward).

## Sonde A — diag_readout_probe.py (le mur du rejeu est dans la LECTURE)

Protocole : rejeu exact du flux [V ; D...D ; V] de retention.retention_curve
(memes distracteurs torch.roll, meme margin=4, meme L1 dernier bloc) sur
frozen_muon/mu_rescale_s[0-9].pt, gaps {0, 2, 4, 8} blocs, n_eval 96,
batch 8. Etages guettes : retrieve_hist, norm_hist (entree/sortie, la
division RMS), attn, update (l'etat), retrieve_out, sortie du modele.

- **P-A1** : le NaN de scoring se REPRODUIT aux gaps >= 4 blocs (256
  latents) sur les bras mu_rescale, et la sortie du modele contient des
  non-finis (ce n'est pas un artefact d'arithmetique de scoring).
- **P-A2 (principale)** : la premiere non-finitude apparait dans la sortie
  de memory.retrieve (lecture MLP d'un etat FINI qui deborde en aval),
  PAS dans l'attention ; au meme moment w1 est fini (etage update sans
  evenement avant le premier evenement de lecture).
- **P-A3** : au premier evenement, l'etat porte une entree enorme mais
  finie (absmax >= 1e15) et le pr du meme passage est < 25% de sa valeur
  gap-0 (l'etat fini, enorme, rang-degenere).
- **P-A4 (branche alternative pre-enregistree)** : si la premiere
  non-finitude est norm_hist_out avec entree FINIE, le coupable est la
  division RMS sur une retrieval quasi nulle (portes erodees) — mecanisme
  « lecture vide », pas « lecture qui deborde ». Les deux branches sont
  distinguables par in_rms_min.
- **P-A5 (controle)** : gaps 0 et 2 — aucun evenement, l1 fini, sur les
  5 graines.

## Sonde B — diag_d2b_train.py (la trace mecanisme de la Figure 2)

Protocole : harnais diag_d2_train a l'identique (memes args que l'echelle
fine de run_d2interne2.sh) + a chaque pas, trace par chunk
(margin.memory_input_stream -> stability.trace_forward, b=0) sur le batch
que le pas vient de voir : theta, lambda_keys_w, lambda_hidden_w,
h0_norm_max, theta*lambda_hidden, w0_spec/w1_spec, finite. Barreaux :
b_t008_s0, b_t008_s1 (frontiere, morts attendues aux pas ~7 et ~2),
b_t005_s0 (controle sous le mur, 300 pas).

- **P-B1** : au pas fatal de t008, la trace par chunk montre theta PLAT
  (variation < x2 sur la trace) pendant que h0_norm_max et
  lambda_hidden_w croissent de plusieurs ordres de grandeur jusqu'au
  premier chunk non fini — la signature D1 que la Figure 2 doit montrer.
- **P-B2** : theta_lambda_hidden franchit ~2 au plus tard au chunk qui
  precede la premiere non-finitude (facteur ~3 d'incertitude proxy,
  comme pour le mur I5).
- **P-B3** : lambda_keys_w reste borne (< 100) tout du long, y compris au
  pas fatal — la branche protegee par la parametrisation ne bouge pas
  (miroir dynamique de la Table 1 v2).
- **P-B4 (controle)** : b_t005_s0 — trace plate sur 300 pas
  (theta_lambda_hidden < 1 partout), aucun chunk non fini, aucune mort.

## Verdicts (26/08, ~15h00 — les deux sondes rendues en 27 min)

### Sonde A — P-A2 TOMBE, et le vrai verdict est plus gros que la prediction

- **P-A1 CONFIRMEE** : nan reproduit (l1 nan aux gaps >= 4 sur 4/5
  artefacts ; evenements "output" non finis presents).
- **P-A2/P-A3/P-A4 TOMBENT — le premier etage non fini est `update`,
  BLOC 1, sur TOUTES les lignes touchees** : w0 ET w1 du bloc 1 non
  finis, biais b0/b1 minuscules et finis. Le mur du rejeu EST une
  explosion d'etat (D1 au rejeu), pas une non-finitude de lecture sur
  etat fini. Le re-diagnostic du 26/08 matin (« readout nan sur etat
  fini rang-degenere ») etait un artefact d'instrument : memory_health
  fait une passe BLOC-0, la chasse (wm_fig2_hunt.py ligne 21) guettait
  blocks[0] uniquement — le bloc 1 etait invisible de tous les
  instruments. health_finite=True disait vrai... du bloc 0.
- **P-A5 partielle** : gap 0 propre 5/5 ; gap 2 propre 4/5 (s2 a des
  evenements des le gap 2, seg 30). s4 est IMMUNISE partout (pr ~53
  plat, zero evenement) — heterogeneite de graines au rejeu, miroir de
  l'entrainement.
- Details : premiers evenements aux segs 30-49 (profond dans le flux),
  dependants du batch (s1 gap4 : premier evt batch 4 ; l1 peut rester
  fini quand l'explosion tombe apres la fenetre scoree).

### Sonde B — P-B1/B2/B3/B4 CONFIRMEES (B1 sur s0 ; nuance s1)

- **P-B1 CONFIRMEE sur b_t008_s0, pas fatal 7** : theta plat
  (0.055-0.060 sur tous les chunks), lambda_hidden 5 -> 221 -> 1.5e3 ->
  5.3e5 (chunks 12-15), h0_norm 0.28 -> 97, w1_spec 1.1 -> 199, etat non
  fini au chunk 16. LA trace de la Figure 2, en main.
- **P-B2 CONFIRMEE** : theta*lambda_hidden = 12.1 au chunk 13, avant la
  premiere non-finitude (chunk 16) ; precurseur des le pas 6 (4.78 au
  chunk 14, l'etat survit — le cliquet inter-pas est visible).
- **P-B3 CONFIRMEE** : lambda_keys plat a ~67 partout, y compris au
  chunk fatal (< 100).
- **P-B4 CONFIRMEE (borne ajustee)** : b_t005_s0, 300 pas, zero chunk
  non fini, theta dans [0.030, 0.038] ; max theta*lambda_hidden = 1.16
  (la prediction disait < 1 : legerement depassee, toujours tres sous le
  mur a 2).
- Nuance s1 : mort au pas 2 reproduite, mais la trace b=0 reste finie
  (max theta*lambda 1.61) — l'element fatal etait une AUTRE ligne du
  batch (la dependance au batch de fig_loss_blind, revue ici). La
  Figure 2 se trace sur s0.
- **Bonus quantitatif** : l'echelle fine en unites theta*lambda encadre
  le seuil theorique : 0.05 vit a max ~1.2 ; 0.08 meurt en franchissant
  ~2-5. Le seuil ~2.37 est DANS l'intervalle.

### Consequence d'instrumentation (a consigner au papier)

Trou d'instrument plus grand que le trou pr-relatif du classifieur : la
sante memoire etait mono-bloc. Toute sonde d'etat doit couvrir TOUS les
blocs a memoire ; c'est le bloc profond (celui qui lit la sortie enrichie
du bloc 0) qui meurt au rejeu.
