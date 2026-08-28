# Predictions — campagne collapse du rang, 2026-08-28

Carte blanche de Josep (28/08) : comprendre et produire des resultats
sur le collapse — video ET texte, plusieurs modeles (le notre,
lucidrains, autres si possible). Priorite confirmee : loi + causal
d'abord, largeur ensuite. Enregistre AVANT tout lancement.

Loi derivee et verifiee en simulation le 27/08
(derivation_loi_pr_2026-08-27.md, 1.2 % median sur 89 configs) :

    1/pr = (terme de compte, ecrit sur les stats des ecritures) + plafond
    forme fermee : 1/pr = (alpha/r_eff).Phi(rho) + 1/d1 + 1/g

Sur artefact reel, l'etat est reinitialise a chaque sequence : le test
est la version FINIE — pr(L) apres L ecritures, predit par la formule a
somme finie sur les statistiques mesurees q_d, p_d des ecritures reelles
(q_d = recouvrement cote cles au lag d, p_d = cote valeurs), sans
parametre libre.

## Volet L — la loi sur les artefacts video geles (sonde
diag_pr_collapse_probe.py, flux cycle 100 clips, carry longue duree,
TOUS les blocs)

- **P-L1** : la trajectoire pr(L) mesuree est predite par la formule a
  somme finie sur les q_d/p_d mesures, a ~15 % pres, sur les bras
  bornes (unit h64, unit10k, usq10k, softclip) — le couplage correctif
  (l'ecriture depend de l'etat) est assez faible pour que le modele
  additif tienne.
- **P-L2** : l'ordre des pr finaux entre bras/graines est reproduit par
  la loi (aucune inversion entre paires bien separees).
- **P-L3** : la video donne des cles fortement correlees (Phi >> 1) et
  un plafond geometrique g bien sous la largeur ; l'etalement 14-53 des
  graines unit se retrouve dans (Phi, g) mesures, PAS dans r_eff.
- **P-L4** : les deux blocs different (le bloc 1 lit un flux transforme
  par le bloc 0) — la sonde multi-bloc est native ici.

## Volet D — dose-reponse alpha-override (mu_unit10k s0/s1,
alpha force a {3e-3, 1e-2, 3e-2} au rejeu)

- **P-D1** : pr decroit quand alpha monte, des montants PREDITS par la
  meme formule avec les memes q_d/p_d (le terme de compte mord quand
  1/alpha descend sous ~Phi.L/r_eff) — un dose-reponse sans parametre
  libre. C'est le test causal du bouton alpha.
- **P-D2** : a l'alpha natif (~1e-4 efface), le terme de compte est
  negligeable sur L <= 800 : la decroissance native est invisible —
  coherent avec pr plat en gap (sonde du 27/08).

## Volet C — bras causal decouple_memory_update (entrainement
mu_unit10k_dec, config EXACTE de mu_unit10k + decouple, graines 0-1)

- **P-C1** : pr plus haut que les graines mu_unit10k appariees, par le
  mecanisme predit : rho_k mesure plus bas (la boucle
  lecture->ecriture qui fabrique de la correlation de cles est coupee)
  et/ou g plus haut.
- **P-C2** : memory_gain pas detruit (dans le bruit de graine de
  mu_unit10k) — le chemin de sortie est inchange par construction.

## Volet X — le texte sur NOTRE banc (cache enwik8 -> embedding fixe
128-d, meme format que egolife ; config mu_unit10k, 4000 pas,
bras unit + usq, graines 0-1)

- **P-X1** : les ecritures texte sont moins correlees que video :
  Phi_texte < Phi_video (les octets successifs n'ont pas la correlation
  des frames).
- **P-X2** : la MEME formule predit pr(L) sur texte a la meme precision
  — la modalite n'entre que par (q_d, p_d, r_eff) mesures, aucune
  correction specifique texte.
- **P-X3 (risquee, qualitative)** : a bras et largeur egaux, pr_texte >
  pr_video si Phi domine ; si c'est g qui domine (la geometrie de
  l'embedding octet est elle-meme etroite), la loi doit quand meme
  suivre — P-X2 est la prediction dure, P-X3 l'attendu directionnel.

## Volet R — pile lucidrains sur texte (repro/titans-pytorch, enwik8)
[apres les volets L/D/C, si la journee le permet]

- **P-R1** : meme loi, memes mesures sur leur NeuralMemory — la loi
  n'est pas un artefact de notre implementation.

## Verdicts (28/08 — campagnes 1 et 2 rendues ; tx10k et volet R en vol)

- **P-L1 : TOMBEE, et remplacee par mieux.** La fermeture par paires
  (validee en simulation) casse sur les flux reels (ecart median 85 %) :
  la correlation longue y est COHERENTE (une ecriture moyenne m
  persistante), ce que la factorisation des 4es moments ne porte pas.
  La loi operationnelle devient la loi DC :
  pr(L) = pr(dec^L W_init + poids.m). ATTENTION : au point final a
  alpha natif elle est une identite (garde-fou consigne dans la note) ;
  ses tests non tautologiques : dose alpha 2-24 % (12 configs, zero
  parametre libre), mi-trajectoire decouple 2 % / texte 10 % /
  cycle-video 24 %, version causale = bon regime (~x2). Les flux
  loop/shuffle cassent la stationnarite de m (108 %) — limite connue.
- **P-L2 (ordre des bras)** : porte par la mesure directe ; l'ordre est
  reproduit par (r_eff, m) — voir P-L3.
- **P-L3 : CONFIRMEE, mecanisme plus fin que prevu.** La video-unit
  donne des ecritures quasi RANG-1 (r_eff 1.6-3.6) dans un sous-espace
  etroit ; l'etalement des graines (pr 1.5-53) est porte par le r_eff
  du flux appris (s4 bloc 1 : r_eff 55 -> pr 49.5 ; ses soeurs
  r_eff ~2 -> pr 1.5-7.3), pas par alpha. usq garde r_eff 16-40 et ne
  collapse pas. NOUVEAU (echelle T2) : le collapse se construit
  PENDANT l'entrainement (unit s1 b0 : pr 69 -> 23 -> 6.6 -> 2.9 de
  2k a 10k pas, r_eff et g s'effondrent avec) ; usq est stable sur le
  meme trajet ; decouple est stable (b1 ~38-48 a tous les checkpoints).
- **P-L4 : CONFIRMEE** : les deux blocs different systematiquement
  (normes, r_eff, geometrie) ; le bloc 1 est souvent le plus atteint.
- **P-D : CONFIRMEE et au-dela.** pr s'ecrase a 1-3 des alpha >= 3e-3
  — et la formule DC le PREDIT a 2-24 % sans parametre libre :
  l'init decroit en dec^L et il reste la fenetre recente dominee par m.
- **P-C1 : CONFIRMEE, effet double a ecrire honnetement.** Bloc 1 :
  ecritures plus fortes (2-3.2) ET plus diverses (r_eff 33-44 vs
  1.8-6) -> pr 35-38 vs 1.5-7.3 (~x10). Bloc 0 : les ecritures se
  taisent (normes ~0.002) — pr haut trivialement (etat = init).
- **P-C2 : PARTIELLE.** memory_gain 0.307/0.320 contre 0.377-0.551
  (5 graines) : sous le min baseline, ~20 % de gain perdu — un vrai
  cout, pas un echange gratuit. mem_effect comparable (0.19-0.20).
- **P-X1 : CONFIRMEE** (avec reserve) : ecritures texte diverses
  (r_eff 12-77), aucun bloc sous pr 13, la derive m existe mais est
  elle-meme de plus haut rang (pr(m) 17-44). Reserves : 4000 pas
  seulement (tx10k a budget video EN VOL) et engagement fonctionnel
  faible (memory_gain 0.02-0.03 contre 0.39 video).
- **P-X2 : CONFIRMEE** au niveau trajectoire DC (10.4 % median).
- **P-X3** : directionnellement vraie a ce budget (pas de collapse
  type unit sur texte a 4k) ; verdict definitif au retour de tx10k.
- **P-R1 : EN VOL** (leur MAC s'entraine ; sonde ecrite).

## Volet 30k — temps longs (decision Josep 28/08 : "joue sur les temps")
Enregistre AVANT lancement. Config identique aux bras 10k, 30 000 pas,
checkpoints tous les 2000, graine 0, une carte par bras.

- **P-T30-1 (unit30k)** : le collapse persiste ou s'approfondit — pr
  bloc 0 a 30k <= sa valeur a 10k (pas de recuperation spontanee) ;
  r_eff du flux reste ~1-4.
- **P-T30-2 (usq30k) — LE test du levier gratuit** : usq reste
  non-collapse a 30k (pr >= 20 et r_eff >= 10 a tous les checkpoints).
  Si usq finit par collapser, la recommandation du papier devient
  "retarde" et non "evite" — a ecrire tel quel.
- **P-T30-3 (dec30k)** : la diversite du bloc 1 persiste (r_eff >= 20,
  pr >= 25) ; les ecritures du bloc 0 restent eteintes.

## Volet S4 — decorrelation du flux (test causal amont)
Cache egolife sous-echantillonne d'un facteur 4 (stride temporel :
frames 4x plus espacees, meme statistique marginale, correlation
successive reduite), bras unit 10k pas, graine 0.

- **P-S4-1** : a config et budget egaux, le flux decorrele donne un
  r_eff d'ecriture plus haut et un plancher de pr plus haut que le bras
  unit natif — la cause amont du collapse video est la correlation
  temporelle du flux, pas la video en soi.

## Volet M — mecanisme du levier usq (sonde enrichie cos_m)
La sonde stocke desormais cos(U_t, m_hat) par ecriture.

- **P-M-1** : dans usq, les ecritures de grande norme sont les moins
  alignees sur la derive (le quartile haut en norme a un |cos_m| moyen
  plus bas que le quartile bas) — la magnitude preservee pondere les
  ecritures INFORMATIVES ; dans unit (normes quasi constantes), aucune
  separation de ce type, et la derive s'accumule a poids egal.

## Regle pr-absolu (classifieur)
Ajout au protocole avant gel : un plancher ABSOLU pr <
0.10 x largeur(W1) (en plus de la regle relative 0.25 x gap-0 et de
cos_ret > 0.95), applique DES le gap 0 — un artefact entraine dans le
collapse est tague collapse d'emblee et la marche de portee s'arrete.
Prediction **P-ABS-1** : le re-scoring des retentions existantes ne
change AUCUN verdict d'arme en tete de table (usq/softclip sains
partout) mais tague collapse des le gap 0 une partie des graines unit
(pr 1.5-13), la ou la regle relative les laissait "saines".

### Verdicts additionnels (28/08 soir)

- **P-ABS-1, premiere moitie RENDUE** : regle implantee dans
  retention.py (COLLAPSE_PR_ABS_FRAC = 0.10 x largeur, appliquee des le
  gap 0 ; pr_width stocke par le builder). Re-scoring des retentions
  existantes (bras 4k, largeur 64, plancher 6.4) : 3 vrais flips
  sain->collapse a mi-marche (clip s0 gaps 8/16 a pr 6.1/5.2 ; unit s1
  gap 32 a pr 6.4) que la regle relative manquait (references gap-0
  basses). Le cas "collapse des le gap 0" vise les bras 10k (largeur
  128, plancher 12.8, graines collapsees a pr 1.5-13.4) — retention
  jamais scoree sur eux : run co-schedule en file (run_ret10k.sh,
  unit10k / usq10k / dec). Ligne limite documentee : la graine unit10k
  s0 b0 a pr 13.4 reste au-dessus du plancher (sensibilite 0.10 vs
  0.125 a ecrire dans le protocole).
- **P-R1, premiere donnee (checkpoint 1000)** : sur la pile lucidrains
  (MAC, enwik8, NS/qk-norm/clipping actifs, tel que configure par leur
  train_mac), les fast-weights des TROIS couches a memoire s'effondrent
  DANS une sequence : pr init 63.7 -> 1.0-3.0 apres ~1150 ecritures,
  r_eff ~ 2, 12/12 flux. La signature unit (ecritures quasi rang-1 +
  crash du pr) se reproduit sur LEUR implementation et sur TEXTE — le
  "texte ne collapse pas" de P-X1 est une propriete de notre bras/banc,
  pas de la modalite. Le pr_dc de cette sonde est tautologique au point
  final (alpha interne chez eux, non modelise) — ne pas le citer ; une
  exception (seq3 mem1 : m ~ 0, etat collapse quand meme) illustre le
  canal fluctuations sans DC. A completer au checkpoint final 4000 :
  la version T2 de leur pile (le collapse s'approfondit-il avec
  l'entrainement ?).

## P-2x2 — cellule manquante direction x magnitude (enregistree AVANT lancement, 28/08 ~17h)

Bras : rawdir_usq_s0/s1 = famille 10k avec la regle de magnitude usq
(psi_c, c=350, theta exterieur) mais direction BRUTE S/||S||_F au lieu de
NS-5(S). Biais inchanges (apparies a usq). 10k pas, frozen-every 2000,
sonde cycle 100 clips. Lancement en file derriere unit30k (carte 0).

Contexte : cellules connues sur bases saines = NS+magnitude-bornee (usq,
tient) et brut+brut (Titans-litteral, 8/8 D1). La cellule
NS+magnitude-brute etait l'hybride rescale — purge, inutilisable. Ce bras
teste si la magnitude SUFFIT.

- **P-2x2-1 (stabilite, confiance 70%)** : 2/2 sains (pas de D1, pas de
  NaN, theta plat). La borne du pas est portee par psi_c seul ; NS n'est
  pas necessaire a la stabilite. Alternative si FAUX : la direction NS
  contribue a casser la boucle de courbure autrement que par la
  magnitude (blanchiment des ecritures) — resultat encore plus
  interessant, a consigner tel quel.
- **P-2x2-2 (rang, confiance 60%)** : collapse du pr AU MOINS aussi dur
  que unit (ecritures brutes quasi rang-1, sans l'inflation de rang de
  NS-5) — r_eff des U_t <= ceux d'usq, pr final <= unit.
- **P-2x2-3 (qualite)** : memory_gain < usq (0.533) — ce que la
  direction NS achete, si elle achete quelque chose, est de la qualite,
  pas de la stabilite. Pas de seuil chiffre, 2 graines seulement.

### Verdict P-ABS-1, seconde moitie (28/08 ~17h, retention absrule 10k)

Fichiers : retention_mu_unit10k_absrule.json / retention_mu_usq10k_absrule.json
(dec en cours). Largeur 128, plancher absolu 12.8.

- **La partie "collapse des le gap 0" est FAUSSE** : au gap 0 les 5
  graines unit10k affichent pr 20.3-71.9, toutes saines. La raison est
  le mecanisme lui-meme : le pr est une fonction de la LONGUEUR de rejeu
  L. La sonde cycle (pr finaux 1.5-7.3) mesure apres ~800 ecritures ; le
  gap 0 de la retention mesure apres UNE presentation — l'etat n'est pas
  encore descendu. J'avais conflate "pr final de sonde" et "pr gap-0".
- **La regle fait exactement son travail a mi-marche** : 7 flips ABS sur
  les graines unit (s3 des le gap 4 a pr 10.1 ; s1/s4 au gap 16 ; s0/s1/
  s3/s4 au gap 32), soit **5/5 graines unit classees collapse au gap 32**
  (s2 via la regle relative/cos a pr 17.0). usq : **0/5, aucun gap,
  aucune graine** — le contraste du levier gratuit existe maintenant dans
  le classifieur OFFICIEL, pas seulement dans la sonde.
- **Le seuil 0.10 track la mort fonctionnelle** : partout ou ABS tire,
  retention_gain <= +0.025, souvent negatif (unit s4 : -0.004/-0.016 ;
  s1 : -0.005). Aucun flip sur une ligne fonctionnellement vivante.
  Bordelines 0.125 : unit s0 gap 16 (14.8), s3 gap 2 (13.7), s4 gap 8
  (15.8) — coherent avec la sensibilite deja documentee.
- **Dose-reponse gratuite** : pr(gap) est monotone decroissant sur unit
  (33->8.8 ; 27->2.9 ; 72->17 ; 20->6.9 ; 26->5.9) et CROISSANT sur usq
  (17->37 au gap 16) — la marche de retention est une rampe de L
  independante qui confirme les deux canaux (descente unit,
  replenissement usq).
- **CAVEAT no-free-lunch, a garder pour le papier** : usq oublie PLUS
  VITE aux petits gaps. ret_gain gap 1 : unit ~0.166 moyen (0.125-0.218)
  vs usq ~0.068 (0.029-0.106) ; gap 2 : unit ~0.13 vs usq ~0.006. usq
  est gratuit au gap 0 (0.467-0.573, egal/superieur a unit) et
  rang-stable partout, mais son horizon de retention est plus court que
  le mi-parcours d'unit. Meme motif que "retention vs memory_gain"
  (bras larges) : le levier est gratuit en memory_gain, pas en horizon.

### Verdict P-X3 (28/08 ~18h, sonde X10k_text, tx10k 10k pas budget apparie)

Largeur 128 (plancher absolu 12.8), 800 ecritures par rejeu, T2 sur
checkpoints 2k-10k, 2 graines x 2 blocs.

- **Le texte ne protege pas, il RALENTIT.** s1 b0 refait unit-video au
  ralenti : r_eff des ecritures 14.7 -> 6.9 le long de l'entrainement,
  pr final 13.2-16.1 a tous les checkpoints — au ras du plancher absolu,
  pente descendante, pas encore classable collapse a 10k. s0 reste
  diverse (r_eff 20-75, pr finaux 25-39), mais avec la MEME derive
  d'entrainement (b0 : r_eff 74.8 a 2k -> 35.0 a 10k).
- La variable operante est la diversite APPRISE du flux d'ecriture, pas
  la modalite : conjugue au volet R (leur stack texte collapse dans une
  sequence), l'enonce "pas de collapse sur texte" du 4k est mort — c'est
  une question de vitesse, la loterie et la derive sont les memes.
- Curiosite consignee : s1 b1 prend le chemin INVERSE (r_eff 50-79,
  m_norm 0.25-0.51, pr qui MONTE 14.5 -> 27.8) — un gros m de haut rang
  qui replenit, coherent avec le canal DC a m divers.

## P-CD / P-R2 — interventions sur le levier de magnitude (enregistrees AVANT lancement, 28/08 ~18h30)

Motivation : le dossier "pallier le collapse" n'a que des contrastes
observationnels entre bras. Ces trois runs le transforment en causal +
transfert. Design commun : decomposer MECANIQUE (la regle appliquee au
rejeu) vs APPRIS (la regle appliquee pendant l'entrainement).

- **P-CD-1 (dose de c au rejeu, notre stack, confiance 65%)** : sur les
  usq10k geles, c-override 35 / 87.5 / 1400 / 3500 au rejeu laisse le pr
  essentiellement PLAT (la diversite directionnelle est apprise ; psi au
  rejeu ne decorrele rien). Si c=35 (psi sature = unit mecanique)
  reproduit le collapse d'unit, le levier est mecanique — consigner.
- **P-R2a (psi_c a l'inference sur LEUR stack, confiance 70%)** : la
  regle usq greffee sur newtonschulz5 de lucidrains, appliquee au rejeu
  seulement sur ckpt_prlaw_final, ne repare PAS le collapse
  intra-sequence (pr reste 1-3 en fin de sequence, r_eff des ecritures
  inchange).
- **P-R2b (psi_c pendant l'entrainement sur LEUR stack, confiance 55% —
  le vrai test de transfert)** : reentrainement 4000 pas identique
  (graine 0) avec la regle active -> le r_eff des ecritures du modele
  final est superieur au baseline et le pr intra-sequence ne descend
  plus a 1-3. Si FAUX : le levier usq est une propriete de NOTRE bras,
  pas une regle portable — a dire tel quel dans le papier (borne la
  section pistes).
  Garde-fou : verifier d'abord que la loss de R2b converge comme le
  baseline (sinon le pr ne compare rien) ; c auto-calibre par EMA de la
  mediane (regle anti-piege-du-c), consigne dans patch_psic.py.

## Campagne week-end 28-31/08 (enregistree AVANT lancement, 28/08 soir)

Contexte : ~160 heures-carte disponibles. Objectifs : (i) rendre la
section (c,alpha) PROSPECTIVE (dose de c a l'entrainement), (ii) passer
les claims 30k et stride de n=1 a n=2, (iii) la baseline lucidrains non
patchee (reference de R2). Configs verifiees identiques a mu_usq10k
(batch 8, T=64, seg 8, dim 128, w1 128x128, qk l2) — seule varie la
quantite testee.

- **P-CDT (dose de c a l'entrainement, usq, c in {35, 87.5, 350 natif,
  1400, 3500}, 2 graines, 10k pas).** psi au depot typique (||S||~350) :
  0.96 / 0.90 / 0.64 / 0.24 / 0.10.
  - P-CDT-1 (confiance 60%) : memory_gain unimodal en c, max au c natif
    350 (la regle de la mediane mesuree) ; les deux extremes perdent
    >= 25%. ALTERNATIVE enregistree : la boucle externe compense le c
    trop grand en gonflant ||S|| (comme l'effacement d'eta_init/alpha) —
    memory_gain plat cote haut ; ce serait un resultat en soi ("c doit
    s'enoncer en unites effectives").
  - P-CDT-2 (confiance 70%) : la stabilite du rang se degrade quand c
    baisse (continuum usq -> unit) : les bras c35 refont unit (pr sonde
    bas, flips retention au gap <= 32 sur >= 1 graine sur 2) ; c1400 et
    c3500 rang-stables.
  - P-CDT-3 (confiance 65%) : en T2 (checkpoints), le r_eff des
    ecritures des bras c35 derive vers le bas comme unit ; c natif
    stable.
- **P-T30-s1** : memes predictions que P-T30-1..3 appliquees a la graine
  1 ; le point est la VARIANCE (loterie unit dej a documentee pr 1.5-49 a
  10k) — aucun verdict 30k ne sera enonce sur n=1.
- **P-S4-s1** : meme direction que P-S4-1 sur la graine 1.
- **P-RLAD-1 (confiance 75%)** : les checkpoints lucidrains 2000/3000/
  final montrent le collapse intra-sequence au moins aussi severe que le
  1000 (pr final 1-3) — l'entrainement ne se repare pas tout seul.
  Reference directe de P-R2a/P-R2b.
- Retentions absrule ajoutees en fin de w2 : familles cdt, rawdir_usq,
  usq30k/unit30k/dec30k (graines presentes), tx10k (cache texte) — le
  classifieur OFFICIEL sur tout le neuf.
