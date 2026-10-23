# Bilan texte : banc gist contre détail pour une mémoire greffée sur un LLM (23/10/2026)

Phase 1 : état de l'art, hôte, générateur, certification, échelle de mémoires non entraînées. Aucun entraînement.
Code et données : `~/work/iclr/text/` sur .135 (carte 3 seulement). Copie locale dans le scratchpad de la session.
Fichiers : `gen_story.py` (générateur), `ladder_text.py` (échelle), `summ_text.py` (mémoire résumé), `analyse_text.py`, `graft_text_smoke.py`, `run_main.sh`. Données : `runs/cert.jsonl`, `runs/cert2.jsonl` (certification, groupes 20000+), `runs/main.jsonl` (mesure, groupes 10000 à 10149), `runs/summ.jsonl`.

## 0. En bref

* Le chemin texte seul marche avec le code de greffe existant, sur le même Qwen2.5-VL-3B, couche 24.
* Huit familles sont certifiées : cinq gist, trois détail. Le plancher vaut 0 par groupe partout.
* Le comptage net (entrées et sorties, ou départs seulement) est **hors de portée de l'hôte 3B** : l'oracle est au hasard. L'exemple « 6 personnes, 2 partent, puis 1 » ne marche que dans sa forme « combien au début ».
* Les mémoires non entraînées les plus simples font presque 0 à D = 1 536 jetons. La meilleure est key (10 à 24 % contrastif). Le seuil à battre est donc bas, et le plafond est haut.
* Un résumé tenu par l'hôte lui même (style ReadAgent) **perd les faits et garde le décor** : 0 à 30 %.
* Limite principale : les faits ressortent trop du décor. Une mémoire entraînée peut apprendre « garder ce qui n'est pas du décor » et tout résoudre, gist et détail. Le test A/B demande donc une v2 : distracteurs de même type et histoires où gist et détail coexistent sous pression de capacité.

## 1. État de l'art

Recalculs et chiffres : sous agent de recherche, sources arXiv. [NV] = non vérifié directement.

### Bancs texte avec fenêtre ou segments

| banc | arXiv | ce qu'il mesure |
|---|---|---|
| BABILong | 2406.10149 | bAbI qa1 à qa20 noyés dans PG19, de 0k à 10M jetons. qa7 = comptage, qa8 = listes. Leaderboard public sur qa1 à qa5. |
| RULER | 2404.06654 | NIAH (single, multi key, multi value, multi query), variable tracking, agrégation (mots fréquents), QA. 4k à 128k. |
| LongBench v1 / v2 | 2308.14508 / 2412.15204 | QA réelles. v1 contient PassageCount (comptage). |
| ∞Bench | 2402.13718 | Au delà de 100k jetons. |
| Michelangelo | 2409.12640 | Latent List : état final après des opérations (suivi d'état). |
| Entity Tracking | 2305.02363 | État de boîtes après opérations, contexte court. |
| SCBench | 2412.10319 | Compression du cache KV : retrieval exact, sémantique, information globale. |
| FlenQA, Lost in the middle, HELMET, NoCha, LooGLE | 2402.14848, 2307.03172, 2410.02694, 2406.16264, 2311.04939 | Raisonnement sous padding, position, évaluation large, livres entiers, dépendances longues. |

### Chiffres de référence (BABILong, moyenne qa1 à qa5, %)

| modèle | 0k | 4k | 16k | 32k | 128k |
|---|---|---|---|---|---|
| ARMT, GPT 2 137M fine tuné | 99 | 98 | 98 | 98 | 97 (1M : 93, 10M : 77) |
| RMT 137M fine tuné | 99 | 92 | 86 | 78 | 59 |
| GPT 4 (0125) | 87 | 74 | 64 | 53 | 36 |
| Qwen2.5-7B-Instruct (recalculé) | 68 | 66 | 63 | 60 | 45 |
| Llama-3.2-3B-Instruct (recalculé) | 57 | 57 | 47 | 38 | n.d. |
| Llama-3.2-1B-Instruct (recalculé) | 41 | 41 | 26 | 16 | n.d. |

Qwen2.5-3B n'est **pas** publié sur BABILong. Titans (2501.00663) donne S-NIAH (MAC : PK 98,4, N 97,4, W 95,2 à 16k) et une seule figure BABILong sans valeurs [NV]. ATLAS (2505.23735) annonce un gain relatif à 10M [NV]. ARMT sur Llama-3.2-1B (2506.05229) : chiffres non extraits [NV].

### Gist contre détail selon la taille mémoire

**Aucun banc public ne fait varier la taille mémoire en séparant gist et détail sur un même récit.** Le plus proche est SCBench : à 1/32 du cache, le retrieval exact s'effondre (57 vers 0,4 avec StreamingLLM) mais l'information globale tient (35 vers 34). Ce « global » n'est pas du comptage d'entités. ReadAgent (2402.09727), ICAE, Gisting, AutoCompressors mesurent la qualité selon le taux de compression, sans typologie gist/détail. Notre protocole serait donc une contribution en soi.

### Contrôle public recommandé

BABILong qa1 à qa5 à 4k et 16k, plus qa7 (comptage) et qa8 (listes), avec notre hôte en pleine attention mesuré par nous. Repères : Llama-3.2-3B 57/47, Qwen2.5-7B 66/63, ARMT 98/98. RULER variable tracking en contrôle secondaire.

## 2. Hôte

**Choix : le même Qwen2.5-VL-3B-Instruct, en entrée texte seule.** Raisons : mêmes poids, même couche de greffe (24 sur 36), même code. La comparaison vidéo contre texte est alors directe : seule la modalité change.

Vérification (`graft_text_smoke.py`, carte 3) :
* identité exacte à l'init : `identical: True, max_abs_diff: 0.0`, couche 24, 512 jetons écrits, mémoire bougée ;
* `up` aléatoire : la greffe déplace les logits (écart max 3,5), tout reste fini ;
* flux en deux morceaux de 256 jetons avec carry : 512 jetons écrits, état conservé.

Point d'attention : `MemoryBranch(2048, 128)` avec les kwargs par défaut de `NeuralMemory` donne des **NaN** sur les états texte de la couche 24. Avec `memory_kwargs_from_args({})` (muon `unit`, alpha 1,27e-4) tout est fini. Il faut toujours passer par cette fonction.

Qwen2.5-3B-Instruct reste l'alternative si l'on veut un hôte purement texte ; je ne l'ai pas mesuré.

## 3. Générateur (`gen_story.py`)

* Un groupe = 4 variantes. Elles partagent tout le remplissage, toutes les positions de faits et la fenêtre. Seul le contenu des phrases de faits change. Les 4 réponses sont distinctes ; chaque lettre sert une fois par groupe (biais de lettre annulé au niveau groupe).
* **Anti copie lexicale** : pour 7 familles sur 9, le sac de mots des 4 variantes est **identique**. Les variantes permutent les mêmes mots sur les mêmes positions ; seule la liaison change (qui porte quelle couleur, quel lieu vient en dernier). Vérifié : la mémoire `lex` est identique entre variantes dans 20/20 groupes pour ces familles.
* Disposition en jetons : [préfixe tronqué pour que le passé fasse exactement P][faits séparés par une phrase de décor][décor de D jetons][fenêtre de W jetons]. Porte de fuite vérifiée par assertion à chaque groupe : fenêtre identique, longueur identique.
* Split : noms, objets, lieux et pièces partagés entre train et eval par hachage salé ; graine salée par split et famille.
* Chaque item porte un QCM (éval) et une réponse libre courte `answer_text` (pour l'entraînement).
* Remplissage : décor sans personnes (5 600 phrases combinatoires). Il exclut tous les mots des familles.
* Réglages : P = 3 072, W = 256, budget B = 384 jetons, D ∈ {0, 1 536} (D = 1 536 = 6 fenêtres).

| famille | type | contenu | sac identique |
|---|---|---|---|
| count_start | gist | 3 à 6 personnes nommées au début ; combien | non (nombre de noms) |
| count_left | gist | 5 personnes, 1 à 4 partent ; combien sont parties | non |
| presence | gist | 4 personnes, 3 partent ; qui reste | oui |
| place | gist | un personnage visite 4 lieux ; où est il à la fin | oui |
| order | gist | 4 événements ; lequel en premier | oui |
| main_object2 | gist | objet perdu, cherché, retrouvé (4 mentions) ; 3 objets incidents | oui |
| colour | détail | 4 personnes, un vêtement coloré chacune | oui |
| number | détail | 4 numéros de casier | oui |
| object_name | détail | 4 objets sur 4 meubles | oui |

## 4. Certification (groupes 20000 à 20014, puis mesure sur 10000 à 10149)

Arrêt précoce à 15 groupes : abandon si oracle groupe < 40 % ou plancher proche du plafond.

| famille abandonnée | oracle groupe (par question) | cause |
|---|---|---|
| count_end (3 entrées ou sorties) | 0 (32) | l'hôte ne calcule pas un effectif net |
| count_end2 (départs seulement, combien restent) | 0 (25 = hasard) | idem : soustraction hors de portée du 3B |
| main_object (pronom « it ») | 33 (80) | ambiguïté objet perdu / objets incidents ; remplacé par main_object2 |

### Grille de certification sur 150 groupes de mesure (contrastif groupe ± SE / brut groupe, par question entre parenthèses)

| famille | oracle | fifo@0 | full@1536 | plancher (blind, empty, other, lex) | verdict |
|---|---|---|---|---|---|
| count_start | 100 / 100 (100) | 100 / 87 (97) | 100 / 84 (96) | 0 | **certifiée** |
| presence | 97 ± 1 / 81 (93) | 76 ± 3 / 43 (72) | 66 ± 4 / 22 (63) | 0 | **certifiée** (oracle) |
| place | 69 ± 4 / 55 (83) | 100 / 97 (99) | 90 ± 3 / 68 (90) | 0 | **certifiée** (fifo@0) |
| order | 100 / 77 (94) | 88 ± 3 / 23 (69) | 95 ± 2 / 31 (77) | 0 | contrastif seulement |
| main_object2 | 100 / 100 (100) | 100 / 100 (100) | 100 / 100 (100) | 0 | **certifiée** |
| count_left | 61 ± 4 / 48 (82) | 65 ± 4 / 21 (73) | 76 ± 3 / 27 (72) | 0 | secondaire (plafond < 80) |
| colour | 100 | 100 | 100 | 0 | **certifiée** |
| number | 100 | 100 | 100 ± 0 | 0 | **certifiée** |
| object_name | 100 | 100 | 100 | 0 | **certifiée** |

Contrôle `other` : la réponse du donneur est choisie à 83 à 100 % ; la cible tombe à 0 par groupe. L'hôte lit bien le contenu de la mémoire.

## 5. Échelle de mémoires non entraînées (150 groupes, contrastif groupe ± SE)

| mémoire | count_start | presence | place | order | main_obj2 | colour | number | object_name | count_left |
|---|---|---|---|---|---|---|---|---|---|
| fifo@0 | 100 | 76 ± 3 | 100 | 88 ± 3 | 100 | 100 | 100 | 100 | 65 ± 4 |
| fifo@1536 | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |
| key@1536 | 10 ± 3 | 0 | 11 ± 3 | 15 ± 3 | **24 ± 3** | 13 ± 3 | 11 ± 3 | 13 ± 3 | 0 |
| mean@1536 | 0 | 0 | 0 | 1 ± 1 | 0 | 0 | 2 ± 1 | 0 | 1 ± 1 |
| lex@1536 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| summ@1536 (n = 10) | 0 | 0 | 0 | 10 ± 10 | 30 ± 15 | 10 ± 9 | 10 ± 10 | 0 | 0 |
| full@1536 (attention) | 100 | 66 ± 4 | 90 ± 3 | 95 ± 2 | 100 | 100 | 100 | 100 | 76 ± 3 |

### Lecture

1. **L'échelle se classe comme prévu par construction.** FIFO fait une marche parfaite (plafond dans le tampon, 0 dehors). key couvre en moyenne 12 % des phrases de faits (B/P = 1/8) et score 10 à 15 % sur les familles à une phrase utile. Les prédictions de conception (lex = 0 quand le sac est identique, key ≈ couverture, fifo en marche) tiennent. Elles n'étaient écrites que dans le code et la docstring, pas dans un fichier séparé.
2. **La redondance est le seul gist que les mémoires naïves gardent.** main_object2 (objet nommé 4 fois) est le meilleur score de key (24) et de summ (30). C'est la version texte de la statistique additive vue en vidéo : une mémoire qui garde « ce qui revient » gagne sur ce gist sans rien comprendre.
3. **Aucun raccourci lexical ni de moyenne.** lex et mean valent 0 sur toutes les familles, y compris count_start où lex liste les noms. Le banc ne se résout pas par mots clés sans liaison. C'est la différence nette avec la vidéo, où mean résolvait hat_color à 99.
4. **Le résumé par l'hôte garde le décor.** Les notes de 3B décrivent le vent et les nuages et oublient les personnes. Le résumé non entraîné n'est pas un sélecteur de gist. Il faut l'entraîner, c'est exactement l'objet de l'A/B.
5. **En pleine attention, le détail tient mieux que le gist intégratif.** À D = 1 536, full garde les détails à 100 mais perd presence (91 vers 66) et count_left (90 vers 76). C'est l'inverse de l'intuition « le détail s'efface d'abord », et un point à rapporter : l'attention retrouve une phrase, elle intègre mal plusieurs phrases distantes.
6. **Score brut contre contrastif.** order brut est à 19 à 31 % par groupe en contexte, contre 93 à 95 en contrastif : biais d'option, comme count en vidéo. Le contrastif par groupe reste la mesure principale ; le brut sert à certifier.

## 6. Seuils à battre (D = 1 536, B = 384 jetons, contrastif groupe, n = 150)

| famille | type | à battre | par | plafond (meilleur de oracle, fifo@0, full@1536) |
|---|---|---|---|---|
| count_start | gist | 10 ± 3 | key | 100 |
| presence | gist | 0 | tous | 97 (oracle) |
| place | gist | 11 ± 3 | key | 100 (fifo@0) |
| order | gist | 15 ± 3 | key | 100 (oracle) |
| main_object2 | gist (redondant) | 24 ± 3 (summ 30 ± 15, n = 10) | key | 100 |
| colour | détail | 13 ± 3 | key | 100 |
| number | détail | 11 ± 3 | key | 100 |
| object_name | détail | 13 ± 3 | key | 100 |

Règle proposée, comme en vidéo : une mémoire « garde » une famille si elle dépasse key de plus de 2 SE (environ 8 points) à D = 1 536. Le contrôle `other` (cible 0, donneur ≥ 83) est obligatoire.

## 7. Coût

* Échelle principale : 57 364 passes en 119 min sur la carte 3 partagée, soit **0,12 s par passe** (contextes de 600 à 3 400 jetons). En vidéo : 0,55 à 1,0 s par passe, plus l'encodeur visuel. Le texte est **5 à 8 fois moins cher par passe**, sans cache d'embeddings à construire.
* Une cellule de 150 groupes (600 passes) coûte environ 1,2 min.
* Le résumé par l'hôte coûte 4,7 min par groupe et famille (génération) : 10 groupes seulement.
* Mémoire GPU : environ 8 Go pour l'échelle.

## 8. Recommandation pour l'entraînement A/B

1. **Avant d'entraîner, une v2 du générateur (≈ 1 h de certification).** Le banc v1 est propre mais facile : les faits sont les seules phrases avec des personnes. Une mémoire entraînée apprendra vite à garder « ce qui n'est pas du décor », gist et détail ensemble, et l'A/B ne séparera rien. Il faut :
   * un remplissage d'**événements distracteurs** (autres personnages, autres objets, autres couleurs, dans d'autres lieux) ;
   * un **mode joint** : une même histoire porte 2 ou 3 faits gist et 2 ou 3 faits détail, variés indépendamment, avec des questions des deux types sur la même histoire ;
   * une **pression de capacité** : plusieurs histoires ou plusieurs épisodes dans un flux, pour que la mémoire ne puisse pas tout garder.
2. **Familles pour l'A/B** : gist = count_start, presence, place, order, main_object2 ; détail = colour, number, object_name. Exclure count_end et count_end2 (l'hôte ne sait pas les faire). count_left en secondaire seulement.
3. **Bras** : A entraîné sur les questions gist seulement, B sur toutes. Prédiction : A ≈ B sur le gist ; A < B sur le détail, avec l'écart qui grandit quand la mémoire rétrécit. Mesurer à D = 1 536 et D = 3 000 avec le contrôle `other`.
4. **Taille mémoire** : partir de la config vidéo (d_mem 128, hidden 64, muon `unit`), puis balayer hidden ∈ {16, 32, 64}. Le contenu factuel d'une histoire v1 tient en 30 à 70 jetons (oracle) ; en mode joint il faut 150 à 300 jetons de faits pour que hidden 16 sature. Pour comparer à budget égal : 384 jetons × 2 048 × 2 octets = 1,5 Mio en tokens, contre 64 Ko pour une mémoire 128 × 64 en fp32.
5. **Contrôle public** : BABILong qa1 à qa5, qa7, qa8 à 4k et 16k, hôte en pleine attention puis greffé.

## Problèmes ouverts

1. Le comptage net est hors de portée de l'hôte 3B. Si l'on veut un gist intégratif fort, il faut soit l'hôte 7B (à certifier), soit une question plus simple (« plus ou moins de personnes qu'au début »).
2. L'oracle de place (faits sans décor) fait moins bien que fifo@0 (69 contre 100). La juxtaposition des phrases de faits gêne l'hôte. Le plafond de place est pris sur fifo@0.
3. presence perd beaucoup en pleine attention avec la distance (66). Le plafond pertinent à D = 1 536 est l'oracle (97).
4. Le résumé par l'hôte n'a que 10 groupes. Le contrôle de mots dans les notes (`notes mention a fact word`) est grossier.
5. Incident : j'ai lancé une fois la grille avec 16,9 Go libres sur la carte 3, sous le seuil de 20 Go, car un processus d'un collègue était revenu entre deux vérifications. Je l'ai arrêté par PID exact au bout d'une minute. Je l'ai relancé derrière une porte qui attend 20 Go libres (`run_main.sh`). La reprise réutilise les lignes déjà écrites.
