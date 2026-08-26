"""Le controle qui manquait a la distillation.

Sur LongVideoBench, une memoire ecrite depuis une autre video donne le meme
score : le gain n'y vient pas du contenu. L'argument qui restait -- << la
distillation, elle, est specifique au contenu par construction >> -- a exactement
la forme de celui que ces controles ont invalide, et n'avait jamais ete teste.
Ces tests pinnent la seule partie qui n'a pas besoin d'un 3B : que le donneur
soit d'un autre jour, et de la meme taille.
"""

from __future__ import annotations

import torch

from titans.graft.eval_distill import donor_span


def _days(sizes):
    return [{"z": torch.zeros(n, 4)} for n in sizes]


def test_the_donor_comes_from_another_day():
    """Un donneur pris dans le meme jour partagerait le decor, les gens et
    l'activite : le controle mesurerait alors la memoire, pas son absence."""
    days = _days([1000, 1200, 900])
    for d in range(3):
        d2, lo, hi = donor_span(days, d, 100, 400)
        assert d2 != d, (d, d2)


def test_the_donor_has_exactly_the_same_number_of_writes():
    """La magnitude de l'etat croit avec le nombre d'ecritures : un donneur plus
    court confondrait << mauvais contenu >> avec << moins ecrit >>."""
    days = _days([1000, 1200, 900])
    for span in (50, 300, 700):
        _, lo, hi = donor_span(days, 0, 10, 10 + span)
        assert hi - lo == span, (span, hi - lo)


def test_the_donor_stays_inside_the_day_it_is_taken_from():
    days = _days([1000, 120, 900])
    d2, lo, hi = donor_span(days, 0, 0, 300)          # jour 1 n'a que 120 jetons
    assert lo >= 0 and hi <= max(hi - lo, 300)
    assert lo == 0, "on ne peut pas commencer apres le debut d'un jour trop court"


def test_the_pairing_is_deterministic():
    """Pas de graine a consigner, pas de derive entre une course et sa reprise."""
    days = _days([1000, 1200, 900])
    assert donor_span(days, 1, 33, 400) == donor_span(days, 1, 33, 400)


def test_the_prompt_survives_having_no_video():
    """Le zero du diagnostic : la question seule, sans un seul jeton visuel.
    Si le professeur est deja prevu de la, la tache est un a priori de langage et
    aucune architecture memoire ne peut y apparaitre."""
    from titans.graft.distill import PROBE, text_only_inputs

    class FakeProc:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert not any(c["type"] == "video"
                           for m in messages for c in m["content"])
            return "<|im_start|>" + messages[0]["content"][0]["text"]

        def __call__(self, text, return_tensors):
            return {"input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])}

    out = text_only_inputs(FakeProc(), PROBE)
    assert set(out) == {"input_ids", "attention_mask"}
    assert "pixel_values_videos" not in out, "aucun jeton visuel ne doit passer"
