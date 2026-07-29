"""Tests for MQAR, the one token-level memory probe kept (shapes + fwd/bwd)."""

import torch

from titans.membench.tasks import associative_recall
from titans.membench.model import RecallModel


def test_task_shapes_and_padding():
    for fn in (lambda: associative_recall(batch=8, num_pairs=4, segment_size=8, seed=0),
               lambda: associative_recall(batch=8, num_pairs=12, segment_size=8, seed=1)):
        b = fn()
        assert b.x.shape == b.y.shape == b.mask.shape
        assert b.x.shape[1] % 8 == 0, "length must be a multiple of segment_size"
        assert b.mask.any(dim=1).all(), "every row must have an answer"
        assert int(b.x.max()) < b.vocab_size
        # targets only where masked
        assert (b.y[~b.mask] == 0).all()


def test_single_answer_per_row_for_recall():
    b = associative_recall(batch=16, num_pairs=5, segment_size=8, seed=1)
    assert (b.mask.sum(dim=1) == 1).all()          # one query answer per row
    assert b.distance.shape == (16,)


def test_model_forward_backward():
    b = associative_recall(batch=8, num_pairs=4, segment_size=8, seed=0)
    model = RecallModel(b.vocab_size, dim=16, depth=2, segment_size=8)
    logits = model(b.x)
    assert logits.shape == (8, b.x.shape[1], b.vocab_size)
    loss = torch.nn.functional.cross_entropy(logits[b.mask], b.y[b.mask])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_eval_seeds_do_not_overlap_the_training_range():
    """Training draws task_fn(step) for step in [0, steps). Eval seeds sitting
    inside that range means the model trains on its own eval batches -- which is
    exactly what the old default range(10_000, 10_020) did for any run past 10k
    steps. Eval seeds must be far beyond any step count we run."""
    import inspect
    from titans.membench import run as run_mod

    for fn in (run_mod.train_recall, run_mod.train_recall_converge):
        default = inspect.signature(fn).parameters["eval_seeds"].default
        assert min(default) >= 1_000_000, (
            f"{fn.__name__}: eval seeds start at {min(default)}, which a long run "
            "would reach during training")
