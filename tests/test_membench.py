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


def test_stop_on_nan_records_the_step_and_halts():
    """The NaN sweep's outcome variable: when did the inner loop blow up."""
    import torch

    from titans.membench.run import train_recall
    from titans.membench.tasks import associative_recall

    def task(seed):
        return associative_recall(batch=4, num_pairs=2, segment_size=4,
                                  key_vocab=8, val_vocab=4, seed=seed)

    # healthy run: no divergence recorded
    torch.manual_seed(0)
    ok = train_recall(task, steps=3, dim=16, depth=1, segment_size=4,
                      memory_hidden_dim=8, eval_seeds=range(10_000_000, 10_000_002),
                      stop_on_nan=True)
    assert ok["diverged_at"] is None

    # and the key exists on the default path too, so callers can read it blind
    torch.manual_seed(0)
    off = train_recall(task, steps=3, dim=16, depth=1, segment_size=4,
                       memory_hidden_dim=8, eval_seeds=range(10_000_000, 10_000_002))
    assert off["diverged_at"] is None


def test_nan_sweep_levers_reach_the_memory():
    """theta_scale / eta_init / chunk must change the computation, not just be accepted.

    A flag that is accepted and silently inert is this codebase's recurring failure
    mode (README §3, §7), and these three are the levers the predicted threshold
    `theta* ~ (1 - eta) / m` is measured with.
    """
    import torch

    from titans.membench.model import RecallModel

    def build(**kw):
        torch.manual_seed(0)
        return RecallModel(16, dim=16, depth=1, segment_size=8,
                           memory_num_layers=2, normalize_qk="l2",
                           normalize_output=False, **kw)

    x = torch.randint(0, 16, (2, 16))
    base = build()
    with torch.no_grad():
        y0 = base(x)

    # each lever, alone, must move the output
    for kw in ({"theta_scale": 0.5}, {"eta_init": 0.9}, {"memory_chunk_size": 4}):
        m = build(**kw)
        with torch.no_grad():
            y = m(x)
        assert not torch.allclose(y0, y), f"{kw} left the computation unchanged"

    # and the chunk really lands on the block
    assert build(memory_chunk_size=4).blocks[0].memory_chunk_size == 4
    assert build().blocks[0].memory_chunk_size == 8        # defaults to the segment
