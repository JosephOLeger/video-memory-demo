# Bilan BABILong : mémoire greffée réentraînée sur des données type BABILong (depuis le 26/10)

Code : `~/work/iclr/text/` sur la tour (et copie sur .135). Générateur `gen_babilong.py`, entraîneur `train_bl.py` (copie de `train_text_ab.py` adaptée), certification `cert_bl.py`, témoin non appris `mean_baseline.py`. Sorties dans `runs_bl/`. Copie locale : `experiments/text/`.

## Synthèse (mise à jour au fil de l'eau)

* **Un oracle qui connaît la question (il n'écrit que les phrases utiles) est une borne de LECTURE, pas une mémoire.** Seuls comptent les résultats avec une écriture qui ignore la question : oracle sur toutes les phrases de faits, ou écriture apprise.
* Borne de lecture (cache kvfull, fins des phrases utiles, 8 cases) : qa1 à qa4 à 100 / 0 par groupe (développement, P 4096 et 8192). Le lecteur sait lire quand la bonne phrase est en mémoire.
* Profond et proj-res, oracle sur tous les jetons de faits (sans connaître la question) : seul qa4 passe (90 / 0) ; qa1, qa2, qa3 et qa7 restent au hasard. Hypothèse : la charge (40 à 280 écritures pour lier personnes et lieux).
* Moyenner les faits (témoin non appris) ne fait rien de plus que la fenêtre seule.
* Pas encore de résultat sur le test officiel avec mémoire.

## Réserves

* Une graine par bras pour l'instant : une piste, pas un résultat.
* L'hôte 3B est faible sur BABILong même avec tout le texte (4k : qa1 82, qa2 43, qa3 36, mesure du 23/10). Le plafond utile est bas sur qa2 et qa3.

## 1. Pourquoi

Josep (26/10) : le banc synthétique est trop pauvre et la mémoire reste spécialisée. On réentraîne sur des faits bAbI cachés dans de vrais livres, avec la même perte (CE en réponse libre, KL vers un professeur qui voit les faits, marge contrastive). La mémoire à 3,3 M de paramètres appris (proj-res) est le bras principal.

## 2. Données d'entraînement (`gen_babilong.py`)

* **Faits** : bAbI en-10k, partie TRAIN seulement. Découpage par question comme BABILong (tous les faits avant la question). Tâches entraînées : qa1, qa2, qa3 (où est X, 1 à 3 faits utiles), qa7 (comptage) et qa4 (relations entre lieux). Tâches tenues à part, jamais entraînées : qa5, qa6, qa8 (évaluées sur le test officiel).
* **Fond** : livres PG19, partie TRAIN (600 livres du premier fragment), validation PG19 pour l'éval de développement.
* **Insertion** : comme `NoiseInjectionDataset` de BABILong (dépôt booydar/babilong) : phrases consécutives d'un livre, faits insérés dans l'ordre à des frontières de phrase tirées au hasard. Différences voulues : on connaît la position de chaque fait ; tous les faits sont dans le passé, au moins D jetons avant la fenêtre ; découpage en phrases par regex (pas de nltk sur la tour).
* **Séparation** : histoires bAbI d'indice multiple de 10 réservées au développement. Les échantillons identiques à un échantillon du TEST bAbI (dont est tiré le test officiel BABILong) sont retirés : qa1 19, qa4 1 480 (histoires de 2 faits, collisions fréquentes), qa7 14, qa2 et qa3 0. Le test officiel BABILong ne sert qu'à l'évaluation finale.
* Les échantillons bAbI de plus de 100 faits ne sont pas tirés (9 % de qa3, aucun ailleurs), pour tenir à P = 2048.

### Versions contrefactuelles (4 par histoire, même fond, mêmes positions, seuls les faits changent)

* qa1 à qa4 : le lieu réponse A est échangé avec un lieu B partout dans les faits (bijection, l'histoire reste cohérente). Les lieux cités dans la question restent fixes.
* **Équilibre de présence** : les 3 lieux B sont des lieux déjà cités dans les faits. Les 4 options apparaissent donc dans chaque version, et savoir quels lieux ont été cités ne donne pas la réponse. Plancher « présence seule » mesuré sur 500 histoires : 25 % par question (le hasard) pour qa1, qa2, qa3.
  * Exception qa4 (2 faits, 2 lieux hors question) : options = les 2 lieux présents et 2 lieux absents. Plancher présence seule : 50 % par question, 25 % par groupe. À lire avec ce plancher.
  * Une première version (échange avec des lieux absents) avait ce raccourci de présence ; elle a été corrigée avant tout entraînement long.
* qa7 : l'acteur de quelques phrases de possession (got, dropped, gave...) est réattribué, avec le minimum de changements et une simulation qui vérifie la cohérence, pour que la personne interrogée porte none, one, two, three objets selon la version.
* Chaque lieu et chaque prénom fait un seul jeton Qwen : les 4 versions ont exactement les mêmes longueurs et positions de faits (vérifié dans l'audit de certification).

### Invites et cibles

* Invite officielle de BABILong pour chaque tâche (instruction, exemples, consigne de format), avec la fenêtre comme contexte pour l'élève.
* Professeur : même invite, avec les phrases bAbI UTILES (supporting facts) avant la fenêtre. Avec toutes les phrases bAbI, il ne répondait juste qu'à 17 % (qa2) et 50 % (qa3) sur les premiers groupes ; avec les phrases utiles, 83 à 100 %. La KL ne s'applique que là où il a juste (métrique officielle).
* Cible de la CE : la réponse du professeur quand il a juste, sinon la réponse au format officiel (« The most recent location of Mary is garden. »).

## 3. Certification (`cert_bl.py`, P 4096, D 300, développement, 16 groupes par tâche, 26/10)

Réponse libre en métrique officielle, par question / vraisemblance des 4 phrases réponses par question / par groupe (meilleure affectation, hasard 4 %).

| tâche | prof. faits utiles | prof. tous les faits bAbI | plein contexte | fenêtre | fond sans faits | notes d'une autre version (suit l'autre) |
|---|---|---|---|---|---|---|
| qa1 | 98 / 100 / 100 | 84 / 84 / 81 | 62 / 64 / 62 | 2 / 25 / 0 | 0 / 25 / 0 | 5 (84) |
| qa2 | 88 / 97 / 100 | 42 / 45 / 50 | 44 / 42 / 19 | 6 / 25 / 0 | 12 / 25 / 0 | 12 (45) |
| qa3 | 95 / 95 / 100 | 39 / 37 / 50 | 31 / 37 / 37 | 17 / 25 / 0 | 17 / 25 / 0 | 12 (37) |
| qa4 | 100 / 100 / 100 | 72 / 73 / 62 | 58 / 58 / 69 | 20 / 25 / 0 | 22 / 25 / 0 | 11 (73) |
| qa7 | 52 / 62 / 81 | 34 / 53 / 87 | 23 / 34 / 62 | 6 / 25 / 0 | 9 / 25 / 0 | 19 (53) |

* Le professeur voit les phrases UTILES : il est juste à 88 à 100 % sur qa1 à qa4. qa7 reste faible (52 %), c'est une limite de l'hôte 3B pour compter.
* Planchers : la fenêtre seule et le fond sans faits ne donnent jamais le groupe (0) ; par question, ils restent au niveau d'un a priori.
* Audit de forme : longueurs et positions des faits identiques entre versions sur 16 groupes sur 16 pour chaque tâche. Aucun raccourci de masse ou de position n'est possible.
* Avec les notes d'une autre version, le professeur suit cette autre version : la réponse vient bien des faits.
* Plafond plein contexte bas sur qa2, qa3 et qa7 : l'hôte se perd dans 4 000 jetons de roman.

## 4. Bras et recette

Même recette que le banc texte (leçons reprises) : flux seulement (`--past-chunk 256 --past-mode slide`), curriculum de longueur P dans {2048, 4096, 8192}, écriture oracle longue puis écriture apprise (`--write-attnpool 8`, curriculum oracle par flux sur 800 pas). Nouvelle base ACP couche 24 ajustée sur ces données (`basis_l24_bl_pca128.pt`, 37 % de variance expliquée sur PG19, contre un texte synthétique bien plus simple avant).

| bras | paramètres appris | état | oracle | écriture apprise | carte |
|---|---|---|---|---|---|
| proj-res (principal) | 3,6 M | 64 Kio | tous les jetons de faits, 2000 pas, depuis zéro | smixA 1500 pas | tour 1 |
| profond de base | 330 k | 64 Kio | tous les jetons de faits, 1500 pas | smixA 1500 pas | tour 0 |
| cache kvfull | | 8 × 2048 | fins des phrases utiles, 1000 pas | smixA 1500 pas | .135 carte 3 |

## 5. Test officiel BABILong

### 5.1 Témoins sans mémoire (hôte Qwen2.5-VL-3B, invite et métrique officielles, 26/10)

Fenêtre = les 256 derniers jetons du contexte. 100 exemples à 4k et 8k, 50 à 16k et 32k.

| tâche | 4k fenêtre / plein | 8k | 16k | 32k |
|---|---|---|---|---|
| qa1 | 26 / 82 | 11 / 82 | 8 / 80 | 2 / 63 |
| qa2 | 25 / 43 | 15 / 43 | 10 / 46 | 16 / 42 |
| qa3 | 34 / 36 | 30 / 32 | 26 / 46 | 26 / 34 |
| qa4 | 25 / 59 | 23 / 64 | | |
| qa5 (tenue à part) | 57 / 77 | 49 / 75 | | |
| qa6 (tenue à part) | 52 / 79 | 48 / 85 | | |
| qa7 | 3 / 27 | 3 / 17 | 0 / 14 | 2 / 26 |
| qa8 (tenue à part) | 30 / 51 | 31 / 56 | | |

* qa1 à 4k redonne exactement la mesure du 23/10 (82 / 26) : la construction de l'invite est validée.
* qa3 : la fenêtre seule fait presque autant que le plein contexte (a priori sur les lieux). Peu de marge.
* qa7 : l'hôte compte très mal, même avec les faits seuls (6 %, voir 5.2). Peu exploitable.

### 5.2 Témoins non appris (`mean_baseline.py`, 4k, faits repérés via le fichier 0k)

| tâche | fenêtre | moyenne des faits (1 jeton doux) | une moyenne par phrase de fait | faits en clair (sélection parfaite) |
|---|---|---|---|---|
| qa1 | 26 | 26 | 26 | 89 |
| qa2 | 25 | 25 | 25 | 35 |
| qa3 | 35 | 35 | 36 | 45 |
| qa4 | 25 | 24 | 27 | 57 |
| qa5 | 58 | 57 | 59 | 86 |
| qa6 | 52 | 53 | 52 | 89 |
| qa7 | 3 | 3 | 2 | 6 |
| qa8 | 30 | 30 | 29 | 58 |

* **Moyenner les faits ne sert à rien** : la moyenne des plongements d'entrée, globale ou par phrase, ne bouge pas de la fenêtre seule.
* Les faits en clair donnent le plafond d'une mémoire parfaite sans compression, avec cet hôte.

## File de nuit (26/10 au soir, machines autonomes)

En cours :
* tour carte 1 : `oracle_deep_ct`, profond, oracle qui ignore la question (jetons de contenu de TOUS les faits : noms, lieux, objets), depuis `oracle_deep`, 1500 pas. Fin vers 19 h 30. Signal précoce (éval 500, 5 groupes seulement) : qa1 80 et 60 / 0, qa3 40 / 0, qa4 80 / 0. À confirmer.
* tour carte 0 : `oracle_projres_ct`, proj-res, même oracle, depuis les cartes linéaires du profond, 2000 pas. Fin vers 20 h 30.
* .135 carte 3 : `oracle_kvfull_se`, cache, oracle qui ignore la question (toutes les fins de phrase de faits, il garde les 8 dernières), 1000 pas. Au hasard à l'éval 500.

En file (tour, worker relancé, cartes 0 et 1 seulement, garde ajoutée) :
1. dès la fin de `oracle_deep_ct` : smoke de 6 pas, puis `smixA_deep_ct` (écriture apprise, 1500 pas, environ 3 h), puis le test officiel (qa1 à qa8 à 4k et 8k, qa1, qa2, qa3, qa7 à 16k et 32k ; mémoire et mémoire d'un autre exemple), environ 2 h ;
2. dès la fin de `oracle_projres_ct` : la même chaîne pour proj-res ;
3. test officiel de l'oracle `oracle_deep_ct` (écriture oracle des faits repérés, n 50), environ 30 min ;
4. graine 1 de `oracle_deep_ct` depuis zéro (2000 pas, environ 3 h).

En file (.135, `runs_bl/q135d.sh`) : après `oracle_kvfull_se`, smoke, `smixA_kvfull_se` 1500 pas (environ 3 h), test officiel 4k et 8k.

Comparaison juste visée : profond et cache avec un oracle qui ignore la question, au même flux d'écriture (tous les faits) et au même état (64 Kio). Le profond reçoit les jetons de contenu, le cache les fins de phrase (ses états bruts gardent la phrase ; dans la base ACP, la fin de phrase a perdu le contenu, rang des clés environ 2).

## Journal

* 26/10 midi : données téléchargées (bAbI 1.2, BABILong test qa1 à qa10 en 0k, 4k, 8k, 16k, 32k, PG19 train fragment 0 et validation). Générateur et entraîneur écrits, smoke test passé (identité à l'initialisation exacte).
* 26/10 12:00 : `smixA_h128` (ancien banc v2, carte 1) arrêté à environ 450 pas sur 1500, raison dans `tower_jobs.log` (direction du 26/10) ; son étape oracle est conservée.
* 26/10 après-midi :
  * deux extinctions de la mémoire (theta de 0,02 à 5e-4, rang des clés 1 à 2). Corrigées par une perte sur les jetons de l'étiquette seulement, et par une base ACP ajustée sur moitié faits, moitié fond (`basis_l24_blf_pca128.pt` : 61 % de la variance des faits gardée, contre 37 %) ;
  * ajout de 3 questions simulées par histoire d'entraînement (simulateur vérifié : 10 000 sur 10 000 bonnes réponses sur qa1 et qa2) ;
  * au plus 2 questions à P 8192, puis 1 pour kvfull, à cause des dépassements mémoire ;
  * smixA_kvfull (issu de l'oracle qui connaît la question) arrêté : 18 % de pas perdus en dépassement mémoire.
* 26/10 12:13 : premier lancement de l'oracle proj-res planté (un qa3 de 1 205 jetons ne tenait pas à P 2048) ; corrigé et relancé avec le générateur équilibré.
