# Perception Test (valid) x Qwen2.5-VL-3B : bilan (arrêté)

Évaluation arrêtée sur décision du lead : 2451 questions notées sur 3018 (échantillon stratifié par tag). Choix 3 voies par logits des lettres. full = clip jusqu'au cut officiel, 2 fps, 640x360 max. window = 4 dernières secondes. blind = texte seul. prior = réponse majoritaire (leave-one-out) des autres questions de même gabarit (texte+options) dans valid.

Marge = full − max(blind, prior). Hasard 33.

| tag | classe | n | full | blind | window | prior | marge |
|---|---|---|---|---|---|---|---|
| distractor object | detail | 152 | 92 | 39 | 85 | 36 | +53 |
| distractor action | other | 138 | 92 | 40 | 90 | 35 | +52 |
| action recognition | other | 243 | 71 | 36 | 69 | 35 | +35 |
| place recognition | other | 126 | 92 | 66 | 91 | 71 | +21 |
| part recognition | detail | 139 | 70 | 44 | 68 | 55 | +15 |
| feature matching | detail | 126 | 69 | 43 | 66 | 57 | +12 |
| containment | other | 133 | 58 | 39 | 50 | 47 | +11 |
| **tout** |  | 2451 | 57 | 44 | 52 | 48 | +9 |
| material | detail | 128 | 91 | 68 | 91 | 83 | +8 |
| pattern discovery | other | 116 | 44 | 36 | 43 | 27 | +8 |
| general knowledge | other | 134 | 51 | 45 | 43 | 33 | +6 |
| action counting | gist | 132 | 48 | 42 | 39 | 42 | +5 |
| conservation | other | 115 | 49 | 43 | 53 | 44 | +4 |
| language | other | 129 | 50 | 46 | 43 | 33 | +4 |
| pattern breaking | other | 135 | 42 | 39 | 35 | 38 | +4 |
| event counting | gist | 138 | 47 | 43 | 34 | 35 | +4 |
| solidity | other | 129 | 51 | 32 | 44 | 48 | +3 |
| colour recognition | detail | 32 | 44 | 41 | 28 | 35 | +3 |
| sequencing | gist | 435 | 49 | 41 | 39 | 46 | +3 |
| motion | other | 157 | 49 | 30 | 39 | 46 | +3 |
| visual discrimination | detail | 143 | 43 | 41 | 36 | 33 | +2 |
| object permanence | gist | 145 | 48 | 32 | 41 | 47 | +1 |
| occlusion | other | 145 | 48 | 32 | 41 | 47 | +1 |
| quantity | other | 96 | 58 | 57 | 56 | 55 | +1 |
| change detection | gist | 127 | 47 | 46 | 44 | 33 | +1 |
| spatial relations | other | 401 | 42 | 42 | 42 | 38 | -0 |
| object counting | gist | 435 | 39 | 37 | 42 | 40 | -1 |
| collision | other | 129 | 58 | 38 | 30 | 60 | -2 |
| state recognition | other | 131 | 57 | 42 | 47 | 60 | -2 |
| object attributes | detail | 195 | 46 | 50 | 36 | 45 | -4 |
| stability | other | 70 | 46 | 50 | 47 | 44 | -4 |
| object recognition | other | 135 | 51 | 37 | 41 | 58 | -6 |
| event recall | gist | 128 | 28 | 35 | 40 | 34 | -7 |
| adversarial action | other | 130 | 50 | 54 | 50 | 57 | -7 |
| task completion | other | 179 | 51 | 55 | 51 | 59 | -8 |

Tags avec marge ≥ 30 points : distractor object, action recognition, distractor action.

Variantes contrefactuelles : 5248 gabarits (texte+options), 1242 répétés, 948 dont la réponse varie selon la vidéo (14380 questions sur 19140). La réponse majoritaire par gabarit donne 70 % sur valid (in-sample) : les priors textuels sont forts.
