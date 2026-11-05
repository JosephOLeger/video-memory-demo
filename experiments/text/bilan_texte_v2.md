# Bilan texte v2 : banc joint, ancre publique, premier A/B et diagnostic de l'écriture (23/10/2026, soir)

Suite de `bilan_texte.md` (v1). Code : `~/work/iclr/text/` sur .135 (carte 3 seulement) et sur la tour (worker à verrou par carte). Copie locale dans le scratchpad de la session.

## Réserve 2 : les familles gist ne sont pas des agrégats (vérifiée le 25/10)

* count_start tient en une seule phrase (« At the beginning of the evening, A, B and C were chatting in the R. »). Elle s'étale sur 18 à 30 jetons, aucune histoire ne dépasse 256. Compter, c'est compter des noms dans une phrase : l'hôte le fait localement.
* order : les 4 incidents s'étalent sur 121 à 1076 jetons (médiane 706), mais le premier est marqué dans sa propre phrase (« At some point », les suivants « Later », « After that », « Then »). Les scores par question ne dépendent pas de l'étalement (kvfull et projres_lencurr en slide : 100 % au delà de 800 jetons).
* main_object : chaque mention de l'objet suffit.
* Il faut donc retirer le cadrage « le profond agrège, le cache garde l'événement ». Lecture correcte : deux faits locaux de nature différente, le profond retient de façon fiable le fait de la première phrase, le cache la phrase marquée, et proj-res fait les deux. Pour tester une vraie agrégation, il faudrait une famille cumulative (par exemple des arrivées réparties sur plus de 512 jetons, sans marqueur ordinal).

## Banc v3 cumulatif (`gen_story3.py`, 25/10)

Trois familles, une phrase par élément, sans marqueur ordinal ni récapitulatif. L'étalement est contrôlé, U[256, 1500] jetons. Les distracteurs sont de même type et placés dans d'autres pièces. Il faut P = 4096.
* count_arrivals : n arrivées (n parmi 2 à 6) ; combien.
* majority : 5 à 9 parapluies, 4 couleurs, une majorité stricte, jamais en première ni en dernière mention ; quelle couleur.
* last_state : un objet posé 3 à 5 fois ; où il est à la fin.

Certification, 20 groupes. Chaque case donne par question en lettre / par question en vraisemblance / par groupe.

| condition | count_arrivals | majority | last_state |
|---|---|---|---|
| oracle | 98 / 86 / 100 | 100 / 100 / 100 | 76 / 42 / 60 |
| sel_best | 50 / 64 / 75 | 98 / 94 / 100 | 51 / 29 / 15 |
| plein contexte D 300 | 35 / 39 / 70 | 88 / 85 / 95 | 31 / 25 / 25 |
| plein contexte D 1500 | 35 / 35 / 75 | 82 / 86 / 95 | 20 / 29 / 15 |
| sel_rand (groupe) | 0 | 25 | 5 à 10 |
| fifo, evfifo, key, mean, lex à D 1500 (groupe) | 0 à 5 | 5 à 20 | 0 à 5 |
| other (contrefactuel) | 0 / 0 / 0 | 0 / 0 / 0 | 10 / 20 / 5 |

* majority est certifiée.
* count_arrivals est certifiée au niveau du groupe. Par question, l'hôte compte mal en plein contexte (35 à 39 %).
* last_state n'est pas certifiée. L'hôte 3B ne repère pas la dernière de plusieurs phrases identiques. Elle est retirée de l'entraînement, gardée à l'éval et notée comme limite de l'hôte.
* **Raccourci de masse (corrigé)** : dans la première version, une variante à n arrivées avait n phrases de fait, donc n fois plus d'écritures. L'oracle profond y atteignait count_arrivals 95 / 0 ; évalué sur histoires équilibrées, il tombe à 5 / 5. Le générateur est maintenant équilibré : mêmes N phrases d'arrivée et mêmes N personnes dans toutes les variantes, les N − n en trop allant dans une pièce d'équilibrage. Le nombre d'écritures est constant à 0,7 % près et décorrélé de n (−0,04). majority était équilibrée par construction (écart de jetons nul).
* Certification équilibrée (vraisemblance par question / groupe) : count_arrivals oracle 76 / 95, plein contexte 35 à 45 par groupe, sel_best 65, planchers 0 à 5 ; majority oracle 99 / 100, plein contexte 95 à 100, sel_rand 50 ; last_state toujours non certifiée.

**Résultats v3** (histoires équilibrées, slide 256, curriculum P ∈ {4096, 8192}, graine 0, 20 groupes, mémoire / autre variante par groupe) :

| mémoire | count_arrivals P 4096 / 8192 | majority P 4096 / 8192 |
|---|---|---|
| kvfull, oracle 8 fins de phrase puis smixA | 15 / 10 | **90 / 0 et 90 / 0** (par question 90 contre 4) |
| kvfull, smixA direct sans oracle | 0 / 0 | 0 / 0 (plat) |
| profond, oracle tous faits (500 pas) | 30 / 0 et 10 / 0 | 10 / 5 |
| profond, smixA | plat, au hasard | plat, au hasard |
| proj-res, oracle puis smixA | plat (coupé au pas 640) | plat |
| **profond, oracle tous faits, 1500 pas** (26/10) | 15 / 0 et 30 / 0 (par question 34 et 35 % contre 22) | **90 / 0 et 90 / 0** (par question 80 contre 4 à 6 %) |
| kvfull graine 1, oracle puis smixA | 20 / 0 et 25 / 0 | 0 et 10 |
| profond, oracle 8 fins de phrase, prolongé à 2000 pas | 15 / 10 et 5 / 0 | 0 / 5 et 5 / 10 (plat) |
| profond, oracle toutes fins de phrase (1000 pas) puis smixA 1500 pas | 5 et 0 | 5 et 0 (plat) |

* **Lecture honnête, corrigée le 26/10** : sur la tâche cumulative certifiée (majority), en streaming, le cache agrège (90, au delà de sel_rand à 50). Le profond n'apprenait pas en 500 pas d'oracle, mais il apprend en 1500 pas (majority 80 / 0 au pas 1250, décollage vers le pas 1000). C'était un artefact de durée. Mais les **charges diffèrent** : le profond reçoit environ 260 écritures pauvres (tous les jetons de faits), le cache 8 états de fin de phrase. Le test à charge égale est en cours : profond en oracle sur les mêmes 8 fins de phrase tirées au hasard, et sur toutes les fins de phrase de faits (environ 15 à 20), 1000 pas puis smixA. S'y ajoutent l'oracle profond sur tous les jetons pendant 1500 pas (.135) et kvfull v3 en graine 1.
* Le raccourci de masse reste documenté comme un piège évité.
* **Loterie de famille chez kvfull** : en v3, la graine 0 apprend majority (90 / 0) et pas count_arrivals (15) ; la graine 1, au pas 1000, apprend count_arrivals (50 / 0 à P 4096, 30 / 0 à P 8192) et pas majority (10). La famille retenue par 8 cases dépend de la graine. C'est cohérent avec la loterie déjà vue en v2 (count_start chez kvfull sur 4 graines sur 8).
* count_arrivals n'est appris par aucune mémoire ; son plafond est bas chez l'hôte.
* Réserves : une graine ; 500 pas d'oracle seulement pour le profond.

## Réserve majeure : préremplissage plein, pas du streaming (relevée le 25/10)

Tous les résultats du banc texte v2 ci dessous ont été obtenus en **préremplissage plein**. L'hôte fait un forward causal complet sur tout le passé (P = 2048, et jusqu'à 8192 pour le curriculum de longueur). La mémoire écrit ensuite les états de la couche 24 issus de ce forward. Chaque état écrit contient donc déjà, par l'attention de l'hôte, de l'information sur tout ce qui le précède.
* L'agrégat (count_start) peut être en partie calculé par l'hôte avant la mémoire.
* Les comparaisons entre mémoires restent justes : toutes voient les mêmes états, à protocole égal. Les graines de confirmation pré enregistrées restent valides à ce titre.
* Ce n'est pas le régime de la greffe vidéo ni celui d'un flux. Le coût croît en O(P²).

Vérification en cours (`--past-chunk W --past-mode chunk|slide`) :
* `chunk` : blocs indépendants de W jetons ;
* `slide` : chaque bloc de W jetons prérempli avec les W précédents comme contexte, en ne gardant que ses propres états ; l'attention couvre au plus 2W.
* (a) Éval seule des checkpoints existants en chunk 256, slide 256 et slide 512, à P 2048 et 8192. La baisse par famille mesure la part que faisait l'attention de l'hôte.
* (b) Réentraînement en slide 256 du profond, de proj-res et de kvfull (graine 0, chaînes complètes). Ce seront les vrais résultats streaming.
* (c) Éval très longue (16k, 32k) en slide seulement, avec le coût par jeton.

### Streaming : premiers résultats en entraînement slide 256

* Lecteur oracle profond en slide sur les 340 jetons de faits : plat sur 500 pas (gap +0,005). Sur les 9 jetons valeur, il apprend : gap +0,31 et order 100 au pas 500. En streaming, la charge empêche l'apprentissage du lecteur.
* smixA entraîné en slide depuis ce lecteur plat (sl_smixA, 1500 pas, P 2048) : count_start 100, main_object 50, order 25, contrôles 0 à 16. C'est autant qu'en préremplissage plein (100 / 45 / 10). Le profond agrège en vrai streaming quand il est entraîné dans ce régime. À P 2048, la chute de (a) venait donc du décalage de distribution. Éval à P 8192 en cours.
* Fine-tune slide (500 pas) des modèles pleins, éval slide 256 à P 2048 : profond 90 / 25 / 40, proj-res 100 / 100 / 75. proj-res s'adapte au streaming sans perte. Évals à P 8192 en cours.
* projres_lencurr évalué en slide 256 sans réentraînement : graine 0 à 100 / 100 / 75 à P 2048 comme à P 8192 ; graine 1 à 100 / 100 / 85 et 100 / 100 / 75. Il se réplique sur 2 graines sur 2. C'est la ligne principale, sur des faits locaux (réserve 2).
* kvfull entraîné en slide (P 2048) : 0 / 100 / 70. Les profils v2 survivent au streaming.
* Le profond entraîné en slide sans curriculum ne porte pas à P 8192 (20 / 5 / 20) : la longueur s'apprend dans les deux régimes.

### (a) Éval seule en blocs, modèles entraînés en préremplissage plein (25/10)

count_start / order / main_object, 20 groupes par affectation. Contrôles à 0 à 5 partout. Détails à 0 à 15 partout.

| modèle | P | plein | chunk 256 | slide 256 | slide 512 |
|---|---|---|---|---|---|
| profond s0 | 2048 | 100 / 10 / 45 | 15 / 5 / 10 | 90 / 5 / 60 | 95 / 10 / 60 |
| profond s0 | 8192 | 5 / – / 35 | 15 / 10 / 5 | 10 / 10 / 20 | 30 / 10 / 20 |
| lencurr s0 | 2048 | 100 / 35 / 35 | 15 / 10 / 40 | 100 / 50 / 40 | 100 / 55 / 40 |
| lencurr s0 | 8192 | 100 / 60 / 35 | 10 / 0 / 15 | 15 / 20 / 25 | 30 / 40 / 55 |
| proj-res s0 | 2048 | 100 / 100 / 70 | 90 / 100 / 65 | 100 / 100 / 65 | 100 / 100 / 65 |
| proj-res s0 | 8192 | | 70 / 95 / 60 | 75 / 100 / 45 | 35 / 100 / 40 |
| projres_lencurr | 2048 | 100 / 100 / 75 | 85 / 95 / 45 | | |
| projres_lencurr | 8192 | 100 / 100 / 85 | 85 / 95 / 50 | | |
| kvfull s0 | 2048 | 0 / 100 / 50 | 0 / 100 / 55 | 0 / 100 / 65 | 0 / 100 / 45 |
| kvfull s0 | 8192 | 0 / 100 / 50 | 0 / 100 / 55 | 5 / 100 / 60 | 10 / 100 / 45 |

* L'agrégat du profond de base dépendait du préremplissage plein : en blocs indépendants, son count_start tombe de 100 à 15, et la portée longue de lencurr chute aussi. La cause reste à trancher par (b) : précalcul par l'hôte, ou décalage de distribution entre les états vus à l'entraînement et ceux écrits en blocs.
* Le cache n'est pas touché : il stocke l'événement au jeton.
* proj-res résiste nettement mieux au découpage. C'est le candidat streaming. Ordre de (b) : profond slide, puis proj-res slide, puis kvfull slide. Par question, ses scores en blocs sont plus bas (count_start 35 à 83 %), mais au-dessus des contrôles.
* Coût d'écriture par histoire : 0,36 s à P 2048 et 1,48 s à P 8192 en slide 256 ; 0,24 et 0,94 s en chunk 256. C'est linéaire en P, et l'état reste de 64 Kio.

## Synthèse pour lundi 26/10 (bilan consolidé du week end, 24/10 au soir)

**Mise à jour du 25/10 au soir (à lire d'abord).**
1. Les familles gist v2 sont des faits locaux (réserve 2). Le cadrage « agrégat contre événement » est retiré : ce sont deux faits locaux de nature différente, la phrase d'ouverture contre la phrase marquée.
2. Tout le v2 est en préremplissage plein (réserve 1). En streaming (slide 256), projres_lencurr tient sans perte : 100 / 100 / 75 à 85 à P 2048 et 8192, sur 2 graines sur 2. C'est la ligne principale pour les faits locaux.
3. Confirmation pré enregistrée de la métrique conjointe : non confirmée (proj-res 2/3, profond 1/3, p = 0,50).
4. Banc v3 cumulatif, certifié pour majority et pour count_arrivals au niveau du groupe. En streaming, kvfull agrège majority (90 / 0, graine 0). **Correction du 26/10, 0 h** : le profond agrège aussi majority quand on l'entraîne assez longtemps. Oracle sur tous les jetons, 1500 pas : gap plat jusqu'au pas 500, décollage vers le pas 1000, majority 90 / 0 à P 4096 et 8192 au final (20 groupes), autant que kvfull. « Le profond n'apprend pas » venait des 500 pas d'oracle. Le raccourci de masse a été trouvé, corrigé et documenté.

**Lecture après les réserves 1 et 2** : dans toute cette synthèse, « agrégat » désigne le fait local de la première phrase (count_start) et « événement » la phrase marquée (order). Aucune famille n'exige d'agréger à travers le flux. Les résultats sont en préremplissage plein sauf mention contraire.

Banc texte v2 : histoires à 8 faits, 3 familles gist entraînées en bras A (count_start, order, main_object), 3 détails (colour, number, object_name), place tenue à part. Hôte Qwen2.5-VL-3B gelé, branche à la couche 24. Mesure par meilleure affectation sur 20 groupes, hasard 4,2 %. Chaque score est contrôlé par la mémoire d'une autre variante, la fenêtre seule et une autre histoire : ils restent à 0 à 15 partout.

**1. La recette phare se réplique sur le gist.** Mémoire profonde h64 (16 576 nombres), écriture apprise (pooling attentionnel par blocs de 8 et curriculum oracle par flux), chaîne complète par graine. count_start est à 80 % ou plus sur 5 graines sur 5 (95 à 100). order 1/5, main_object 0/5 (20 à 55). Les détails restent au hasard dans tous les bras.

**2. À budget égal, profond et cache ont des biais de profil, pas une dissociation.** kvfull : 8 états bruts de 2048, lecture apprise, écriture Perceiver. 8 graines contre 8 :

| famille | profond (≥ 80) | kvfull (≥ 80) | Fisher | Mann-Whitney exact |
|---|---|---|---|---|
| count_start | 100, 95, 95, 100, 95, 95, 100, 100 → 8/8 | 0, 0, 100, 5, 0, 100, 100, 100 → 4/8 | 0,077 | 0,39 |
| order | 10, 65, 100, 20, 0, 100, 95, 100 → 4/8 | 100, 100, 100, 100, 100, 70, 100, 100 → 7/8 | 0,28 | 0,044 |
| main_object | 0/8 (20 à 60) | 1/8 (30 à 80) | 1 | |
| conjoint (count_start ET order) | 4/8 | 3/8 | 1 | |

Le profond agrège à chaque graine, le cache une graine sur deux, de façon bimodale. Le cache rend order plus régulièrement. À 5 contre 5 l'écart semblait net (Fisher 0,048). Il s'est réduit avec les graines 5 à 7.

**3. Le proj-res est le seul bras fiable sur les deux familles.** Lecteur profond plus MLP résiduels de largeur 1024 sur k, v, q et up (3,3 M de paramètres appris), 5 graines : count_start 5/5, order 5/5, main_object 70, 75, 5, 35 et 50 (0/5). À état égal et paramètres égaux, le même ajout sur le cache (kvfull-proj, 3 graines) donne count_start 5, échec (gap plat) et 0, et order 100, échec et 100. count_start : proj-res 5/5 contre kvfull-proj 0/3, Fisher p = 0,018. Avec les mêmes MLP, seul l'état profond donne l'agrégat. L'oracle de kvfull-proj s1 donnait number 85, mais cela n'a pas survécu à l'écriture apprise.

*Métrique conjointe, choisie après coup* : une graine compte si count_start ET order sont à 80 % ou plus. Au moment du choix (graines 0 à 6) : proj-res 5/5, profond 3/7 (s2, s5, s6), kvfull 2/7 (s2, s6), Fisher proj-res contre kvfull p = 0,028, contre profond p = 0,081. Avec les graines 7 (toutes deux 100 / 100) : profond 4/8, kvfull 3/8. Cette métrique a été définie après avoir vu les scores. C'est une tendance à confirmer sur de nouvelles graines, pas une preuve.

**Confirmation pré enregistrée (écrite le 25/10 vers 3 h, avant tout lancement).**
* Hypothèse : le proj-res atteint la métrique conjointe (count_start ET order à 80 % ou plus, éval finale au pas 1500, 20 groupes, meilleure affectation) sur plus de graines que le profond de base.
* Graines nouvelles, jamais vues : proj-res s5, s6, s7 et profond s8, s9, s10. Chaînes complètes, recette identique aux graines 0 à 4 (lecteur oracle sN, puis proj-res oracle 1000 pas, puis smixA 1500 pas ; ou lecteur oracle sN, puis smixA 1500 pas).
* Test : Fisher exact unilatéral sur les seules graines nouvelles, proj-res > profond, seuil 0,05. Avec 3 contre 3, seul 3/3 contre 0/3 franchit le seuil (p = 0,05). Le résultat sera donc aussi rapporté en Fisher unilatéral sur toutes les graines (proj-res s0 à s7 contre profond s0 à s10), en le signalant comme analyse mixte.
* Aucune graine ne sera écartée ni relancée. Un run qui plante est compté comme un échec, sauf panne matérielle documentée.
* **Résultat final** : proj-res s5 échec (100 / 0), s6 réussite (100 / 100), s7 réussite (100 / 100) ; profond s8 échec (100 / 40), s9 réussite (100 / 100), s10 échec (95 / 0). proj-res 2/3 contre profond 1/3, Fisher unilatéral p = 0,50. **La confirmation est négative** : l'avantage conjoint de proj-res vu sur les graines 0 à 4 ne se reproduit pas sur graines nouvelles. Ce qui tient : count_start est à 80 % ou plus sur toutes les graines de toutes les mémoires profondes ; order reste une loterie de graine.

**4. Le curriculum de longueur répare la portée.** smixA_s0, entraîné à P 2048, tombe à 5 % sur count_start à P 8192. Avec P tiré dans 2048, 4096 et 8192 : count_start 100 aux trois longueurs. Une graine. order et main_object restent à 35 à 60.

**5. Pourquoi les détails ne passent pas : l'interférence au stockage.** C'est la chaîne de diagnostic du week end. Chaque test écarte une cause.
* *Pas la lecture.* Nourrie des seuls 9 jetons valeur, la même mémoire lit les détails (object_name 75, main_object 80, colour 35). Sous environ 340 écritures (tous les jetons de faits), ils tombent à 0 à 10. La charge gouverne.
* *Pas la taille de l'état.* h512 (8 fois plus d'état) : rien ne change. Avec la réserve qu'il est sous réglé.
* *Pas le rang des clés.* Il est bas après entraînement (11 directions effectives pour 342 faits). Mais le relever par pénalité de décorrélation (55) ou par blanchiment ZCA (69) ne fait passer aucun détail.
* *Pas l'adressage.* Sans aide, la lecture vise mal les détails : rang de la clé visée 0,80 à 0,91, pour 0,5 au hasard. Avec un objectif InfoNCE, elle vise juste (rang 0,003 à 0,006 sur colour, number et object_name), count_start et order montent à 100. Mais les détails restent à 5 à 10.
* *C'est le stockage.* On lit la mémoire à la clé écrite elle même, et on compare à la mémoire d'une autre histoire à même charge. La part propre à l'histoire vaut 0,1 à 0,4 en cos à 9 écritures. À 340, elle tombe à 0,01 à 0,06. La lecture rend alors surtout une carte k→v générique, portée par M_0 appris : cos de 0,7 à 0,86 pour sa propre mémoire comme pour une autre. Pour les détails, la part propre reste de 0,015 à 0,035, même avec InfoNCE.
* *Le retard d'élan est réel mais marginal.* Avec eta ≈ 0,998, les dernières écritures sont encore dans l'élan S. Un flush à la lecture (environ 220 pas à gradient nul, alpha à 0) les révèle : le dernier décile passe de 0,009 à 0,089. Mais il dégrade le début du profil, et il fait baisser l'éval (order 90 → 55, main_object 55 → 25). Le lecteur est co adapté à l'état non vidé. Ni le flush ni l'override d'eta (0,99 ou 0,9) ne font passer un détail.

**6. Ce qui reste ouvert pour lundi.**
* Le levier pour les détails n'est ni la lecture ni l'adressage. C'est la capacité effective d'écriture sous charge : moins d'écritures (sélection), ou une règle d'écriture qui protège les anciennes associations.
* L'écriture apprise actuelle donne une lecture presque indépendante de la question : cos entre lectures 0,83 pour smixA_s0. Le gist y passe par l'agrégat, pas par un rappel adressé.
* Fiabilité 8 contre 8 (profond, kvfull) et 5 graines proj-res : en cours, résultats dimanche.

## 0. En bref (état au 23/10)

* **Résultat principal (24/10, 00:40)** : bras A, écriture apprise (pooling attentionnel et curriculum oracle par flux), pas 1500, 20 groupes. count_start est à **100 %** par groupe contre 0 avec la mémoire d'une autre variante, d'une autre histoire ou la fenêtre seule (hasard 4,2 %). main_object est à 45 contre 0. Une graine. La base pool entraînée n'apprend pas, donc la comparaison « deep contre cache » reste ouverte.

* **Banc v2 certifié** sur 6 familles en mode joint : count_start, order, main_object (gist) ; colour, number, object_name (détail). Chaque histoire porte les 8 faits à la fois, au milieu de distracteurs de même type. Plancher à 0 partout.
* presence échoue dès qu'il y a des distracteurs (0 par groupe, même en contexte). place n'est certifiée que sur l'oracle (93) ; elle tombe à 67 à budget égal.
* **Ancre publique** BABILong 4k, notre hôte en plein contexte : qa1 82, qa2 43, qa3 36. Fenêtre seule : 26, 25, 35.
* **Le premier A/B (recette vidéo) n'apprend rien.** Après 250 pas, la mémoire a zéro contenu sur toutes les familles.
* **La cause est l'écriture.** En texte, l'information d'un fait reste sur le jeton valeur. Le stride 8 ne l'écrit qu'une fois sur 8, et l'ACP 128 détruit la moyenne de 8 jetons. Une sonde linéaire le chiffre : couleur 100 % sur le jeton valeur, 36 % sur la moyenne projetée, 17 % sur le jeton du stride.
* **La lecture apprend quand le fait est écrit.** Avec une écriture oracle (jetons des faits seuls), count_start atteint 80 % par question au pas 500, contre 10 % avec la mémoire d'une autre variante.
* **Une écriture apprise donne le premier contenu.** Au pas 1500, bras B : count_start 50 % par groupe, main_object 35 %, contre 0 avec la mémoire d'une autre variante. Les détails restent au hasard.
* **Elle commence aussi à sélectionner.** Pooling attentionnel avec curriculum oracle par flux : poids de pooling gist 1.16, détail 1.10, distracteur 1.08, décor 0.94 ; signal faible en éval au pas 1000 (count_start 30 % par affectation contre 0 pour l'autre variante). Avec lr ×30 sur la tête : décor 0.86, événements 1.2 à 1.3.
* **Le bras C (distillation sans question) n'éteint pas la mémoire.** Il en fait une mémoire de **continuation** : KL pondérée 0.38 avec la mémoire contre 3.35 avec la fenêtre seule, soit 89 % de la perte dépendante du passé. Mais elle ne sert à rien sur les faits interrogés (0 %). C'est un résultat en soi : l'objectif définit ce qui est gardé.

## 1. Générateur v2 (`gen_story2.py`)

* **Mode joint** : une histoire porte les 8 faits interrogés, 5 gist et 3 détail. Chaque famille a sa propre permutation variante vers réponse, donc chaque famille forme un groupe contrefactuel sur les mêmes 4 histoires. Une même histoire sert les questions gist et détail ; c'est ce qu'exige l'A/B.
* **Distracteurs de même type, partout** : autres personnes qui discutent, entrent et sortent d'AUTRES pièces ; vêtements colorés ; numéros de casier ; objets sur des meubles ; voyageurs ; incidents ; objets perdus. Ils apparaissent dans l'épisode des faits, avant, après et dans la fenêtre. Ils sont identiques entre variantes et ne sont jamais interrogés. Chaque question nomme une entité que seul le module de fait utilise, donc aucune ambiguïté.
* **Épisodes et distance** : [épisodes de distracteurs][épisode des faits, environ 1 200 jetons][épisodes de distracteurs, environ D jetons][fenêtre]. P = 8 192, D ∈ {0, 3 000, 6 000}.
* **Sac de mots** identique entre variantes pour toutes les familles sauf count_start (nombre de noms) et main_object (l'objet perdu est nommé 4 fois).

### Information par histoire contre budget

| contenu (moyenne, 30 histoires) | jetons |
|---|---|
| faits interrogés | 341 |
| distracteurs de même type | 2 793 |
| décor | 5 058 |
| épisode des faits | 1 217 |

Les 8 réponses font 16 bits. Ce n'est pas la capacité brute qui limite, c'est la sélection. Budget de l'échelle : 384 jetons de texte, qui couvrent les faits mais seulement 12 % des événements. Mémoire profonde de l'A/B (d_mem 128, hidden 64) : 16 576 nombres, soit 64,75 Kio fp32. C'est 0,9 jeton de KV (bf16, 36 Kio par jeton chez Qwen2.5-3B) ou 8,1 jetons d'états cachés.

## 2. Certification v2 (15 groupes, 20000 et suivants)

Contrastif groupe / brut groupe (par question). Plancher = blind, empty, other, lex, mean : 0 partout.

| famille | oracle | sel_best (budget égal) | ep (épisode) | full@0 | full@6000 | verdict |
|---|---|---|---|---|---|---|
| count_start | 100 / 100 (100) | 100 / 93 (98) | 100 / 100 | 93 / 93 | 87 / 0 (37) | certifiée |
| order | 100 / 73 (92) | 93 / 73 (92) | 67 / 67 | 100 / 87 | 87 / 33 | certifiée (contrastif) |
| main_object | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | certifiée |
| colour | 100 / 100 | 100 / 87 | 100 / 100 | 93 / 93 | 93 / 93 | certifiée |
| number | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | certifiée |
| object_name | 100 / 100 | 100 / 93 | 100 / 93 | 80 / 67 | 73 / 33 | certifiée |
| place | 93 / 93 (98) | 67 / 60 | 47 / 33 | 47 / 27 | 13 / 7 | oracle seulement |
| presence | 87 / 53 (87) | **0 / 0 (23)** | 0 / 0 (2) | 0 / 0 (2) | 0 / 0 | **non certifiée** |

Contrôle `other` : donneur choisi à 87 à 100 %, cible 0.

Mémoires naïves à D = 6 000 (contrastif groupe) : fifo 0, evfifo 0 (« garder ce qui n'est pas du décor » ne sert plus à rien), mean 0, lex 0, key 0 à 20, sel_rand 0 à 27 (main_object 60 par redondance).

**Enseignant en réponse libre** (faits visibles, réponse normalisée, 60 histoires train) : count_start 100, order 95, main_object 100, colour 100, number 100, object_name 100. presence 70 (45 à 50 pendant l'entraînement), place 57. presence est donc retirée de l'entraînement.

## 3. Ancre publique : BABILong 4k

100 exemples par tâche, prompts et métrique officiels (`babilong_ext/`), hôte Qwen2.5-VL-3B en texte.

| | qa1 | qa2 | qa3 |
|---|---|---|---|
| plein contexte | **82** | 43 | 36 |
| fenêtre seule (256 jetons) | 26 | 25 | 35 |
| repère publié Llama-3.2-3B, 4k | 80 | n.d. | n.d. |
| repère publié ARMT 137M fine tuné, moyenne qa1 à qa5 | 98 | | |

qa3 (trois faits chaînés) est déjà au niveau de la fenêtre seule en plein contexte : l'hôte 3B ne le résout pas.

## 4. Échelle v2 avec la métrique de l'A/B (60 groupes)

Score identique à celui des mémoires entraînées : vraisemblance des 4 options dans le format réponse libre. Cellule = par question % / groupe par meilleure affectation % (hasard 4 %, ±SE bootstrap environ 2 à 7). n = 60 groupes. Les conditions sans information donnent 25 / 0 à 10.

| condition | count_start | order | main_object | place | presence | colour | number | object_name |
|---|---|---|---|---|---|---|---|---|
| oracle | 100 / 100 | 95 / 100 | 100 / 100 | 74 / 98 | 56 / 98 | 100 / 100 | 100 / 100 | 100 / 100 |
| sel_best (384 jetons) | 93 / 100 | 70 / 90 | 100 / 100 | 57 / 60 | 24 / 22 | 100 / 100 | 100 / 100 | 99 / 100 |
| full@0 | 96 / 100 | 68 / 88 | 100 / 100 | 41 / 38 | 12 / 0 | 100 / 100 | 100 / 100 | 90 / 98 |
| full@6000 | 83 / 100 | 42 / 43 | 100 / 100 | 15 / 2 | 18 / 0 | 100 / 100 | 100 / 100 | 73 / 80 |
| fifo@0 | 25 / 2 | 5 / 0 | 30 / 22 | 85 / 90 | 10 / 0 | 42 / 33 | 51 / 41 | 39 / 35 |
| fifo@3000 et @6000 | 25 / 0 à 10 partout | | | | | | | |
| evfifo@0 | 25 / 2 | 26 / 33 | 98 / 98 | 69 / 78 | 8 / 0 | 89 / 88 | 89 / 88 | 73 / 76 |
| evfifo@6000 | 25 / 4 | 25 / 2 | 25 / 6 | 25 / 6 | 25 / 2 | 25 / 8 | 25 / 4 | 25 / 4 |
| sel_rand@6000 | 32 / 12 | 34 / 23 | 49 / 33 | 32 / 20 | 26 / 2 | 34 / 20 | 27 / 8 | 25 / 12 |
| key@6000 | 30 / 12 | 26 / 4 | 38 / 19 | 27 / 15 | 24 / 6 | 26 / 8 | 28 / 10 | 24 / 12 |
| mean@6000 | 25 / 6 | 22 / 2 | 24 / 8 | 23 / 4 | 26 / 4 | 26 / 4 | 28 / 2 | 23 / 0 |
| lex@6000 | 25 / 6 | 25 / 2 | 25 / 6 | 24 / 12 | 26 / 4 | 24 / 2 | 25 / 0 | 26 / 2 |
| other (donneur) | 0 / 0 | 1 / 0 | 0 / 0 | 9 / 0 | 14 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Lecture :
* Plafond à budget égal (sel_best) : 90 à 100 sur les 6 familles d'entraînement. Plancher (mean, lex, key, fifo et evfifo à distance) : 0 à 12. L'écart à combler est donc entier.
* Le plein contexte à 6 000 jetons garde les détails et count_start, mais perd order (43) et place (2). Une mémoire peut battre l'hôte lui même sur ces familles.
* sel_rand (384 jetons au hasard) reste sous 30 : il faut **sélectionner**, pas seulement stocker.
* presence n'existe que sur l'oracle (98) ; elle reste en éval seulement. place est la famille tenue à part ; son plafond réaliste est 60.
* Avec ce score, count_start plein contexte passe de 0 (QCM lettres brut) à 100 : l'ancien 0 venait du format lettres, pas du contenu.

## 5. A/B texte, premier passage (recette vidéo)

* Bras : A = count_start, order, main_object ; B = A + colour, number, object_name. 3 questions par histoire, les deux variantes de la paire servent à chaque pas.
* Famille tenue à part : place. Justification : certifiée sur l'oracle, du suivi d'état comme order, et un vocabulaire (lieux) absent des familles entraînées. presence est en éval seulement.
* Perte : CE sur la réponse libre ; KL vers l'enseignant (faits visibles) seulement si sa réponse libre normalisée est juste, avec son format comme cible ; relu(1 + CE_propre − CE_autre).
* Histoires neuves à chaque pas, D tiré dans [0, 6 000], P = 8 192, W = 256. Écriture stride 8, chunk 8, lr 1e-3, mémoire unit h64.
* Coût : 9,7 s par pas, réparti en encodage 1,2 s, écriture 2,4 s, rétropropagation de l'écriture 4 s, lectures et enseignant. Pic 10 à 11 Gio.

**Résultat : à plat.** A_h64_s0 au pas 250 : 0 sur toutes les familles, donneur 25 %, écart CE_autre − CE_propre ≈ 0 depuis le début. A_h16 pareil sur 220 pas. Les deux sont coupés, PID exacts, raison notée. Les runs à stride 8 suivants (poolA_s0 jusqu'à 1 000 pas, diag P 2048) confirment le plat.

## 6. Diagnostic : l'écriture est le goulot

### Sonde linéaire sur les états couche 24 (300 histoires, complet 2048 / ACP 128)

| ce qui serait écrit | couleur (8 classes) | dizaine du nombre (10) |
|---|---|---|
| jeton valeur | 100 / 100 | 100 / 89 |
| moyenne des 8 jetons du bloc | 92 / **36** | 60 / 23 |
| jeton gardé par stride 8 | 24 / 17 | 24 / 20 |
| « . » final | 67 / 37 | 36 / 27 |

En texte, la valeur ne vit que sur son jeton. Le stride la saute. La moyenne la garde en 2048 dimensions mais l'ACP figée la perd. Josep avait raison sur le goulot d'entrée.

### Écart d'entraînement et éval (P 2048, D ≤ 600)

| run | écriture | lr | écart CE_autre − CE_propre (blocs de 100 pas) | éval |
|---|---|---|---|---|
| diag s8 | stride 8 | 1e-3 | +0.00, −0.02 | 0 |
| **oracle-write** | jetons des faits seuls | 3e-3 | +0.00, +0.02, +0.12, +0.30, **+1.01** (pas 500) | **count_start 80 % par question, autre variante 10 %, fenêtre 25 %** |
| wap8 (pooling attentionnel appris) | blocs de 8 | 3e-3 | ≈ 0 jusqu'à 500 | hasard |
| wap8 init lecteur oracle | | 3e-3 | ≈ 0 jusqu'à 260 | hasard |
| wap8 + curriculum par bloc | | 3e-3 | ≈ 0 | coupé (la moyenne des faits d'un bloc perd la valeur) |
| **wap8 + curriculum par flux** (p de 1 à 0 sur 800 pas, init lecteur oracle) | | 3e-3 | +0.54, +0.37, … puis **+0.18, +0.10, +0.29 à p = 0** | pas 1000 : count_start 30 par affectation. **Pas 1500 (20 groupes), mem / autre : count_start 50 / 0, main_object 35 / 0, order 15 / 0** ; détails au hasard ; écart remonte à +0.71 |
| idem, lr ×30 sur la tête de pooling (tour) | | 3e-3 | +0.56, +0.26, +0.20, +0.38, +0.08, +0.12, +0.19 | pas 1000 : count_start 10 / 0, le reste au hasard |
| wp8 + down appris (tour) | moyenne de 8 | 3e-3 | ≈ 0 jusqu'à 490 | coupé |
| stride 1 + chunk 64 (tour, smoke 300 pas, init oracle) | tous les jetons | 3e-3 | +0.09, +0.02, −0.02 ; |g| 130 à 400, toujours écrêté ; pr(W1) 34 à 37, theta 0.022, pas de NaN | pas 300 : hasard. Stable mais à plat ; 15 s par pas, pic 18 Gio |

**Poids du pooling appris** (k × poids, 1 = uniforme), à p = 0 :
* curriculum par flux, bras B : gist 1.16, détail 1.10, distracteur 1.08, décor 0.94 ;
* avec lr ×30 : décor 0.86, distracteur 1.21, gist 1.22, détail 1.31.

C'est le mécanisme de sélection qu'on veut comparer entre A et B.

### Métrique

* Le QCM par lettres n'est jamais entraîné. Avec |delta|/|h| entre 0.7 et 1.0 il ne mesure plus rien : 0 même quand le contenu est là.
* La mesure principale devient la vraisemblance des 4 options dans le format entraîné (« Answer with a few words. »), notée par question et par groupe selon la **meilleure affectation** des 4 réponses aux 4 variantes (hasard 1/24).
* Le contrastif centré par option, repris de la vidéo, échoue sur les réponses ordinales : count_start oracle donne 80 % par question mais 0 % en contrastif, car la mémoire préfère aussi les comptes voisins.

## 7. Bras C : mémoire de continuation

C_h64_s0 (1 000 pas, P 8192, stride 8) :
* KL pondérée par w_t : 3.35 avec la fenêtre seule, 0.38 avec la mémoire au pas 990 (mem_gain 3.0 nats).
* |delta|/|h| de 0 à 0.42.
* Norme relative d'écriture de 6.3 à 4.2.
* eta de 0.995 à 0.991, alpha de 1.2e-4 à 2.2e-4, pr(W1) de 39 à 49.
* **La mémoire n'est pas éteinte, elle est très utilisée.**

QCM lettres à l'éval 1000 (100 groupes, D 3000 et 6000) : 0 à 4 % partout, réponse libre 0 à 1,5 %. La mémoire de C transporte ce qui revient dans la suite du texte (noms et lieux des distracteurs) et dégrade la réponse aux questions. **L'objectif définit ce qui est gardé.** Ré-éval avec la nouvelle métrique (60 groupes, D 3000 et 6000) : par affectation 1,7 à 10 % partout, autant que l'autre variante et la fenêtre (hasard 4,2 %). **Aucun contenu de fait, confirmé.**

## 8. Ce qui tourne et la suite

* File de la tour : ré-éval de C (nouvelle métrique), résumé de l'hôte, puis **smixA_s0** (bras A de la recette qui marche), puis s1c64 A et B (1 000 pas chacun), puis pool B et fifo B appariés au stride 1. A_h64_s0_long (stride 8) est en #HOLD.
* La comparaison A contre B se lit sur smix : B est fait, A tourne ensuite. Les graines 1 et 2 viennent après.
* .135 carte 3 : smix jusqu'à 1 500 pas, puis l'écriture oracle poursuivie jusqu'à 1 500 pas (borne de la sélection parfaite).
* Mis en attente (#HOLD) dans la file de la tour jusqu'à une écriture qui marche : unit/usq (B, h64 et h16, 3 graines), proj, h512, d512, sweep, poolB, fifoA. Les lancer avec stride 8 reproduirait le plat.
* Réglage actuel : `unit`. usq = `unit_squash`, c = 1 (comme la greffe vidéo).

### Bras A avec la recette smix (smixA_s0)

Par affectation, sur 10 groupes, mem / autre variante / fenêtre / autre histoire :

| pas | main_object | count_start | order | détails (non entraînés) |
|---|---|---|---|---|
| 500 | 50 / 0 / 10 / 0 | 10 / 0 / 0 / 11 | 0 | 0 à 10 |
| 750 | **60 / 0 / 10 / 0** | 20 / 10 / 0 / 0 | 10 / 0 / 0 / 0 | 0 à 10 |
| 1000 | 60 / 0 / 10 / 0 | 20 / 0 / 0 / 0 | 0 / 10 / 0 / 0 | 0 à 10 |
| **1500 (20 groupes)** | **45 / 0 / 5 / 0** | **100 / 0 / 0 / 0** (par question 81 contre 9, réponse libre 85) | 10 / 0 / 0 / 5 | 0 à 10 |

Écart d'entraînement +1.9 au pas 750 et +2.3 au pas 1000, contre +0.5 à +0.7 pour B. pr(W1) 41,5, eta 0,998 : pas d'effondrement. À ce stade, A (gist seul) apprend mieux le gist que B (count_start 100 contre 50). C'est une graine par bras. Poids du pooling : gist 1.07, détail 1.06, distracteur 1.02, décor 0.97.

Graine 1 (chaîne complète) : le lecteur oracle s1 garde un écart plat jusqu'au pas 500, alors que la graine 0 était à +1.0. Son éval 500 donne pourtant count_start 35 / 5 et order 15 / 0. La loterie des graines est déjà visible au niveau du lecteur.

### Base pool entraînée : n'apprend pas

poolB (même recette, 16 384 nombres) : écart plat au lecteur oracle (500 pas) et au smix (1 320 pas, ±0.01). Éval au hasard. Ce n'est pas encore « deep > cache ». SlotMemory n'apprend pas dans cette recette, même avec l'écriture oracle. Il faut un réglage propre au pool (lr, tau_init) avant toute conclusion. fifoB est en cours.

### Bases entraînées appariées

* pool (SlotMemory, 128 cases × 128 = 16 384 nombres, contre 16 576 pour le profond) et fifo (P = 1, les 128 derniers blocs de 8, soit 1 024 jetons). Même recette : oracle-write 500 pas, puis curriculum par flux 1 500 pas.
* Limite : en oracle-write, pool ne garde que les 256 derniers jetons de faits sur environ 340 (128 cases de 2). Le profond écrit tous les jetons de faits.
* Graines : smixB_s1 refait toute la chaîne en graine 1 (oracle-write s1, puis smix s1). La variante qui partage le lecteur oracle s0 ne passe que si une carte reste libre.
* Chaque éval note aussi la mémoire d'une autre histoire (lk_ostory). Hasard de la meilleure affectation : 1/24 = 4,2 %.

## 10. Week end 24 et 25/10 : réplique, bases, diagnostics

Format : groupe par meilleure affectation, 20 groupes, mem / autre variante / fenêtre / autre histoire. Hasard 4,2 %. Réponse libre = réponse gloutonne juste, normalisée.

### Réplique de la recette phare (smix, bras A, chaîne complète par graine)

| run | count_start | main_object | order | réponse libre count_start | pr(W1) |
|---|---|---|---|---|---|
| smixA_s0 (unit) | 100 / 0 / 0 / 0 | 45 / 0 / 5 / 0 | 10 | 85 % | 41,5 |
| smixA_s1 (unit) | 95 / 0 / 0 / 0 | 55 / 5 / 5 / 0 | 65 / 0 | 90 % | 42,7 |
| smixA_s2 (unit) | 95 / 0 / 0 / 0 | 20 / 5 | **100 / 0 / 0 / 5** | 70 % (order 100 %) | 43,6 |
| smixA_usq (s0) | 80 / 0 / 0 / 0 | 55 / 0 / 5 / 0 | 15 | 64 % | 42,9 |
| smixB_s0 (unit) | 50 / 0 | 35 / 0 | 15 | | |
| smixB_s1_full (unit) | 95 / 0 / 0 / 0 | 20 / 5 / 5 / 0 | 5 | 66 % | 40,5 |

* Le gist est retenu à chaque graine, avec 0 pour la mémoire d'une autre variante et pour celle d'une autre histoire. Les détails restent au hasard dans tous les bras. place (tenue à part) est à 0.
* unit et usq se valent à ce budget. Aucun effondrement de rang.
* Réponses libres brutes gardées : `runs_ab/keep/` sur la tour.

### Passé plus long que l'entraînement (smixA_s0, éval seule)

| P | count_start | main_object |
|---|---|---|
| 2048 (entraîné) | 100 | 45 |
| 4096 | 20 (D 300), 40 (D 2000) | 35, 0 |
| 8192 | 5 partout | 35 (D 300), 10, 5 |

Le contenu ne se transfère pas à un passé plus long.

**Curriculum de longueur** (smixA_s0_lencurr : P tiré dans 2048, 4096, 8192 à chaque pas, init smixA_s0, 1500 pas, D 300, 20 histoires) :

| P | count_start | order | main_object |
|---|---|---|---|
| 2048 | 100 | 35 | 35 |
| 4096 | 100 | 45 | 35 |
| 8192 | 100 | 60 | 35 |

* Le curriculum répare la portée longue du gist : count_start passe de 5 à 100 à P 8192. Contrôles (autre variante, fenêtre seule, autre histoire) à 0 à 5.
* Une seule graine. order et main_object restent plus bas que smixA_s0 à P 2048 (45). Les détails restent au hasard.

### Bases de cache entraînées

* SlotMemory (ACP 128, 128 cases) n'apprend dans aucun réglage avec tous les jetons de faits. Six configurations ont été testées (lr, tau, tête ×10, clés centrées).
* **Ce n'est pas un bug.** Nourri des 8 jetons valeur (un par fait), le même pool lit les détails : order 100 / 0, object_name 50 / 0, main_object 45 / 0. Dans un autre run, colour 90 / 0, avec une sonde couleur à 96,5 %. Il lit donc au jeton, mais il sert 1 à 3 familles selon le run.
* Le profond fait le contraire. Sur tous les jetons de faits, il agrège le gist : sonde count_start 97 % à 1500 pas, couleur et nombre au hasard. Nourri des 8 jetons valeur, il apprend lentement (500 pas : order 20, object_name 20).
* La figure de complémentarité attend le profond à 1500 pas sur les mêmes jetons. Le repérage des nombres écrit désormais tous les jetons chiffres : Qwen découpe 71 en « 7 » et « 1 ».
* **kvfull, la base de cache juste à budget égal.** 8 états bruts × 2048 = 16 384 nombres, sans ACP, lecture apprise. L'écriture apprise est de style Perceiver : 8 requêtes apprises sur tous les jetons du passé. Même chaîne que le profond (oracle sur les jetons valeur, puis curriculum par flux, bras A, 1500 pas).

  Fiabilité = nombre de graines à 80 % ou plus (chaînes complètes, bras A, pas 1500). Test exact de Fisher bilatéral.

  | famille | profond, 5 graines | kvfull, 5 graines | p |
  |---|---|---|---|
  | count_start | 100, 95, 95, 100, 95 → 5/5 | 0, 0, 100, 5, 0 → 1/5 | 0,048 |
  | order | 10, 65, 100, 20, 0 → 1/5 | 100, 100, 100, 100, 100 → 5/5 | 0,048 |
  | main_object | 45, 55, 20, 45, 45 → 0/5 | 40, 30, 50, 60, 50 → 0/5 | 1 |
  | détails, place, presence | 0 à 15 → 0/5 | 0 à 20 → 0/5 | 1 |

  * L'autre variante et l'autre histoire restent à 0 à 10 partout, pour les deux mémoires.
  * L'écriture Perceiver peut en principe agréger, puisque chaque requête voit tous les jetons. La graine 2 le fait (count_start 100). L'échec n'est donc pas imposé par construction, ce qui rend la comparaison juste.
  * À budget égal, le profond retient l'agrégat de façon fiable, et le cache l'événement en une phrase. Pas de « profond > cache » global : deux profils opposés. C'est significatif au seuil nominal (p = 0,048, n = 5). Avec Bonferroni sur deux tests, p = 0,096.
  * Test de Mann-Whitney exact sur les scores bruts (5 contre 5, bilatéral, par permutation, ex aequo comptés pour moitié) : count_start p = 0,071, order p = 0,048, main_object 0,63, colour 0,37, object_name 0,44, number 1. Sur les scores bruts, count_start ne passe pas 0,05, parce que la graine 2 de kvfull (100) égale le profond. Le profil « order pour le cache » est le plus net des deux.
* fifo : même lecture que le pool, arrêté.

### Pourquoi le profond ne lit pas les détails : la charge

Le profond est nourri en oracle des seuls jetons valeur, environ 9 écritures par histoire (bras B, 1500 pas depuis zéro). On le compare à la borne, qui écrit tous les jetons de faits, environ 340.

| famille | 9 jetons valeur | environ 340 jetons de faits |
|---|---|---|
| order | 100 | 80 |
| main_object | 80 | 30 |
| object_name | 75 | 0 à 10 |
| count_start | 70 | 100 |
| colour | 35 (sonde 67 %) | 0 à 10 (sonde 12 %) |

* La lecture du profond sait extraire un détail. Ce qui l'en empêche, c'est la charge.
* Peu d'écritures donnent les détails. Beaucoup d'écritures donnent l'agrégat (sonde count_start 97 %).
* **Capacité ou superposition : test h512.** 8 fois plus d'état (131 712 nombres, 514 Kio) sur les mêmes environ 340 jetons de faits, 1500 pas. Résultat : count_start 100, order 95, main_object 5, object_name 15, colour 5. Sonde couleur 13,5 %, sonde count_start 95,5 %.
* Multiplier l'état par 8 ne débloque rien. Réduire les écritures de 340 à 9 débloque les détails. Le goulot est donc l'interférence entre écritures sous charge, pas la taille de l'état ni la largeur des projections (proj-res).
* **Réserve sur h512** : il est sous utilisé. pr(W1) vaut 41 sur 128, contre 32 à 35 sur 64 pour h64, et son theta médian est plus petit (0,0073 contre 0,0104). lr et theta étaient pensés pour h64. h512 est donc un indice, pas une preuve.
* **Rang des clés écrites** (100 histoires, checkpoints existants) :

  | checkpoint | écritures | rang eff. des clés | \|cos\| moyen | pr(W1) |
  |---|---|---|---|---|
  | h64, tous les faits | 342 | 10,8 | 0,23 | 31,6 / 64 |
  | h64, jetons valeur | 9 | 4,1 | 0,23 | 14,1 / 64 |
  | h512, tous les faits | 342 | 9,1 | 0,24 | 41,2 / 128 |

  342 faits sont projetés sur environ 11 directions effectives de clés. Plus d'état ne change pas ce rang, qui est fixé par k_proj et par les états de l'hôte. L'agrégat passe, les faits individuels interfèrent. Le rang des clés est bas, mais le relever ne suffit pas (test ci dessous).
* **Test de décorrélation des clés** (même setup que la borne, pénalité λ·‖corr(K) − I‖² avec λ = 1) : le rang effectif des clés monte à 55 (contre 11), et pr(W1) reste à 33. count_start 100, order 15, main_object 5, colour 5, object_name 0. Les détails ne passent pas, et order et main_object reculent. Le faible rang des clés n'est donc pas la cause. Ce qui reste établi, c'est que la charge gouverne (340 contre 9 écritures). À l'initialisation, le rang des clés vaut environ 35, et l'entraînement normal le fait tomber à 11.
* **Blanchiment ZCA des clés et des requêtes** (même setup, statistiques glissantes des clés écrites) : le rang effectif des clés monte à 69 (pic 74), pr(W1) 31. count_start 100, main_object 15, order 5, colour 5, number 10, object_name 0. Négatif comme la pénalité. Élargir le rang des clés ne suffit pas, par aucune des deux voies.
* number reste bas même avec tous les jetons chiffres. Le problème est propre aux nombres.

### Adressage : la lecture vise-t-elle la clé du fait ?

Mesures sur 60 histoires : on lit la même mémoire avec les 8 questions. Le rang de la clé visée est cos(q, k_t) de la clé du jeton valeur parmi les clés écrites. 0 = meilleure, 0,5 = hasard, et au delà de 0,5 la clé est anti visée. C'est un proxy, puisque la mémoire est un MLP.

| mémoire | écritures | cos entre lectures | variance question / histoire | rang de la clé visée |
|---|---|---|---|---|
| borne oracle | 341 | 0,54 | 65 % / 19 % | main_object 0,02, place 0,04, colour 0,80, number 0,88, count_start 0,91 |
| oracle 9 jetons valeur | 9 | 0,48 | 61 % / 7 % | colour 0,01, object_name 0,00, order 0,00, count_start 0,00 |
| smixA_s0 | 256 | 0,83 | 33 % / 54 % | main_object 0,02, le reste 0,46 à 0,81 |
| smixA_projres | 256 | 0,69 | 23 % / 49 % | main_object 0,02, place 0,05, order 0,98 |

* À 9 écritures, q vise la bonne clé, et les détails passent.
* À 340 écritures, la lecture dépend de la question, mais elle vise de travers sur les détails.
* Avec l'écriture apprise, la lecture est presque un sac indépendant de la question. Le gist passe alors par l'agrégat, pas par un rappel adressé.
* Levier testé : un objectif d'adressage InfoNCE (`--addr-nce λ`). q au dernier jeton de la question doit choisir la clé du jeton valeur visé parmi les clés écrites, avec une température apprise. Borne oracle, h64, 1500 pas depuis zéro, λ = 0,3 choisi par smoke. Au smoke, le rang de la clé visée tombe de 0,38 à 0,01 en 20 pas, avec λ = 0,1 comme avec λ = 1, sans dégrader la CE. Critère de succès : rang < 0,1 sur colour et object_name à l'éval addr-stats, détails qui passent, count_start conservé. Si c'est le cas, smixA + InfoNCE suit.

  **Résultat (oraclewB_nce, 1500 pas) : l'adressage est réparé, les détails ne passent pas.**

  | famille | rang de la clé visée, borne | rang, InfoNCE | score, borne | score, InfoNCE |
  |---|---|---|---|---|
  | colour | 0,80 | 0,006 | 0 à 10 | 10 |
  | number | 0,88 | 0,003 | 0 à 10 | 5 |
  | object_name | | 0,005 | 0 à 10 | 5 |
  | count_start | 0,91 | 0,00 | 100 | 100 |
  | order | | 0,00 | 80 | 100 |
  | main_object | 0,02 | 0,013 | 30 | 30 |

  * Le cos moyen entre lectures passe de 0,54 à 0,25. Contrôles à 0 à 15.
  * q vise désormais la bonne clé, mais la mémoire MLP ne rend pas la valeur à cet endroit. Après environ 340 écritures, le contenu est écrasé par les écritures voisines. Le goulot n'est pas l'adressage, c'est le stockage sous charge.
  * Critère non rempli : smixA + InfoNCE n'est pas lancé. Réserve : le rang est un proxy cosinus. Le test direct serait l'auto lecture à la clé écrite k_t, comparée à v_t.

### Lecture à la clé écrite : le stockage est le goulot

On lit chaque mémoire à ses propres clés écrites k_t, et on compare à la même lecture dans la mémoire d'une autre histoire à même charge (M_other). Cela neutralise la triche par une application k→v générique. Mesures sur 20 histoires (`--key-read`).

Spécificité = cos(M_own(k_t), v_t) − cos(M_other(k_t), v_t) :

| famille | 9 écritures | 340, borne | 340, InfoNCE |
|---|---|---|---|
| order | 0,42 | 0,06 | 0,16 |
| object_name | 0,33 | 0,035 | 0,035 |
| main_object | 0,31 | 0,034 | 0,10 |
| colour | 0,08 | 0,014 | 0,020 |
| number | 0,03 | 0,029 | 0,015 |
| count_start | 0,09 | 0,010 | −0,002 |

* À 340 écritures, la lecture rend surtout un vecteur indépendant de l'histoire : cos de 0,7 à 0,86, que la mémoire soit la sienne ou une autre. La part propre est de 0,01 à 0,06, contre 0,1 à 0,4 à 9 écritures.
* InfoNCE relève la part propre du gist (order, main_object), pas celle des détails.
* Identification de v_t parmi 20 histoires (hasard 0,05). À 9 écritures, le témoin M_other est au hasard : main_object 0,85, object_name 0,60. À 340, le témoin monte à 0,1 à 0,9, ce qui montre que M_0 appris porte une carte générique. L'écart propre − autre vaut +0,75 pour main_object et +0,2 à +0,3 pour les détails sur la borne. Bruit d'environ ±0,1 à n = 20.
* Profil par position d'écriture (borne, déciles) : 0,012 au début, 0,05 au milieu, 0,009 à la fin. Les premières écritures sont écrasées, mais les dernières sont aussi faibles. Ce n'est donc pas une récence pure. InfoNCE a la même forme.
* **Flush de l'élan et override d'eta** (borne 340, éval seule). Le flush applique N écritures à gradient nul avec les portes médianes de l'histoire, par le vrai pas de la mémoire. alpha est mis à 0 pendant le flush, avec une variante alpha médian pour référence. Spécificité par décile, du début à la fin :

  | lecture | décile 0 | 2 | 4 | 6 | 8 | 9 |
  |---|---|---|---|---|---|---|
  | sans flush | 0,012 | 0,049 | 0,043 | 0,038 | 0,017 | 0,009 |
  | flush 10 | 0,011 | 0,046 | 0,043 | 0,035 | 0,024 | 0,013 |
  | flush 50 | 0,009 | 0,037 | 0,037 | 0,037 | 0,034 | 0,028 |
  | flush auto (233 pas) | −0,001 | 0,010 | 0,036 | 0,038 | 0,070 | 0,089 |
  | flush auto, alpha médian | 0,002 | 0,011 | 0,031 | 0,039 | 0,069 | 0,089 |
  | eta 0,99 | 0,011 | 0,044 | 0,044 | 0,038 | 0,022 | 0,006 |
  | eta 0,9 | 0,017 | 0,025 | 0,040 | 0,025 | 0,023 | 0,012 |

  Éval (20 groupes, par affectation) :

  | lecture | count_start | order | main_object | détails |
  |---|---|---|---|---|
  | sans flush | 100 | 90 | 55 | 0 à 5 |
  | flush 10 | 100 | 90 | 45 | 0 à 10 |
  | flush 50 | 100 | 85 | 40 | 0 à 10 |
  | flush auto | 100 | 55 | 25 | 0 à 5 |
  | flush auto, alpha médian | 100 | 45 | 30 | 0 à 15 |
  | eta 0,99 | 100 | 75 | 35 | 5 à 15 |
  | eta 0,9 | 100 (par question 75) | 60 | 40 | 0 |

  * Les dernières écritures sont encore dans l'élan : le flush complet fait passer le dernier décile de 0,009 à 0,089, soit le maximum de toutes les mesures à 340.
  * Mais en mode unit, chaque pas ajoute theta·unit(S) dans la même direction. Le flush complet pousse W loin, efface le début du profil et dégrade l'éval. Le lecteur est co adapté à l'état non vidé.
  * alpha à 0 ou médian donne le même résultat (alpha ≈ 1e-4).
  * Aucun réglage de dynamique à la lecture ne fait passer un détail. Le retard d'intégration explique la fin du U inversé, pas la perte des détails.
* Conclusion : la lecture peut viser juste, mais sous charge le contenu individuel n'est pas dans les poids. Le goulot est l'interférence au stockage. C'est cohérent avec les négatifs de la décorrélation, du blanchiment et de h512.

### proj-residual en oracle (1000 pas depuis le lecteur profond au pas 500)

| famille | profond (borne) | proj-res, `down` ×0,01 | proj-res, `down` gelé |
|---|---|---|---|
| count_start | 100 | 100 | 100 |
| order | 80 | 100 | 100 |
| main_object | 30 | 50 | 60 |
| détails | 0 à 10 | 0 à 15 | 0 à 15 |

Les MLP gagnent 20 à 30 points sur le gist en un jeton, sans rien changer aux détails. C'est cohérent avec un goulot de charge plutôt que de largeur.

### Bras proj (apprentissage de `down` et des MLP)

* proj1024 depuis zéro (3,3 M de paramètres) : plat pendant 1000 pas, coupé.
* proj-residual (applications linéaires du profond et MLP à zéro, `down` à lr ×0,1) : `down` dérive et détruit le lecteur, coupé. Relancé avec `down` à lr ×0,01 et avec `down` gelé.

### Incidents du week end

* Repérage du jeton valeur faux dans sa première version (articles, nombres, mots multiples). Les runs concernés ont été coupés et relancés.
* Le résumé de l'hôte attend toujours .135.

## 9. Incidents

* Un lancement sous la porte des 20 Go (v1, voir `bilan_texte.md`).
* Runs coupés par moi, PID exacts, raison notée dans `tower_jobs.log` ou ici : A_h64_s0, A_h16_s0, diag s1 (17 s par pas), wp8, wp8td, wap8 seul, wap8 init oracle, curriculum par bloc, oracle-write au pas 520 (repris de son checkpoint).
* Le résumé de l'hôte (summ_text2) plante sur la tour : RoPE de Qwen2.5-VL en generate par batch, selon la version de transformers. Il attend .135, sans priorité.
* smixB_s1 à lecteur partagé tué au démarrage (PID exact), remis en fin de file.
* Le worker de la tour a été redémarré une fois : ouverture du verrou root en lecture, et porte à 15 Go sur la carte 1 à la demande du coordinateur.

**Charge à durée égale (26/10, 3 h).** L'oracle profond sur 8 fins de phrase reste plat à 2000 pas, alors que l'oracle sur tous les jetons donne majority 90 / 0 à 1500 pas. Le profond n'est donc pas gêné par la charge forte. Il apprend avec des écritures denses et n'apprend pas avec 8 écritures, au moins dans ce budget. kvfull fait l'inverse : il apprend avec 8 fins de phrase. Une graine par bras ; la graine 1 de l'oracle dense tourne. Le smixA chaîné sur l'oracle 8 fins de phrase a été coupé (PID exact, raison dans tower_jobs.log) : sa source est plate.

**Écriture apprise et graine 1 (26/10, 5 h).**
* smixA 1500 pas sur l'oracle dense de la graine 0 : majority 90 / 0 à P 4096 et 95 / 0 à P 8192 (par question 78 et 68 % contre 6 et 14 %). L'écriture apprise garde majority. count_arrivals 10 et 20 / 0.
* Oracle dense, graine 1, 1500 pas : il décolle lui aussi vers le pas 1000, mais sur l'autre famille. count_arrivals 30 / 5 et 30 / 0 (par question 38 et 43 % contre 18) ; majority 10 / 10 et 5 / 0.
* Le profond a donc la même loterie de famille que kvfull : une seule des deux familles décolle, et laquelle dépend de la graine. Ce qui se répète sur 2 graines sur 2 : le décollage vers le pas 1000 avec écritures denses.
* En file sur la tour : smixA sur l'oracle de la graine 1 (v3_smixA_1500_s1), puis la chaîne h128.
