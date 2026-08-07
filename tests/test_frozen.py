"""The frozen artifact: config travels with the weights, and loading is strict."""

import pytest
import torch

from titans.worldmodel.frozen import build_cfg, load_frozen, save_frozen
from titans.worldmodel.model import WorldModel


def _model(**cfg):
    torch.manual_seed(0)
    return WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                      memory_num_layers=2, **cfg)


def test_roundtrip_reproduces_the_architecture_and_the_outputs(tmp_path):
    cfg = build_cfg("sane_ns")
    m = _model(**cfg)
    p = save_frozen(m, cfg, str(tmp_path / "f.pt"), meta={"label": "x"})

    back, info = load_frozen(p)
    z = torch.randn(2, 8, 8)
    m.eval()
    with torch.no_grad():
        ref = m(z)
    # Not bit-exact: the loaded model runs with grad disabled everywhere, and the
    # inner loop's vmap'd gradients pick different kernels with and without autograd
    # (measured ~3e-6 here). A config mismatch would be orders of magnitude larger,
    # and the strict load already rules out a different parameter set.
    assert torch.allclose(ref, back(z), atol=1e-4, rtol=1e-4)
    assert info["meta"]["label"] == "x"
    assert info["arch"] == {"dim": 8, "depth": 1, "segment_size": 4}


def test_loaded_model_is_frozen_and_in_eval(tmp_path):
    cfg = build_cfg("sane_ns")
    p = save_frozen(_model(**cfg), cfg, str(tmp_path / "f.pt"))
    back, _ = load_frozen(p)
    assert not back.training
    assert not any(q.requires_grad for q in back.parameters())


def test_cascade_config_survives_the_roundtrip(tmp_path):
    """The six cascade flags are exactly what a plain checkpoint cannot carry."""
    cfg = build_cfg("sane_ns", cascade=True, reservoirs=5, leak="surprise",
                    leak_rate=0.01, readout="gated", selective=True)
    m = _model(**cfg)
    p = save_frozen(m, cfg, str(tmp_path / "c.pt"))
    back, info = load_frozen(p)
    assert info["cfg"]["memory_use_cascade"] is True
    assert info["cfg"]["cascade_reservoirs"] == 5
    assert info["cfg"]["cascade_leak"] == "surprise"
    assert info["cfg"]["cascade_leak_rate"] == 0.01
    assert info["cfg"]["cascade_selective"] is True
    assert back.blocks[0].memory.use_cascade_momentum


def test_a_plain_checkpoint_is_refused_with_a_pointer(tmp_path):
    p = tmp_path / "plain.pt"
    torch.save({"model": _model().state_dict(), "step": 10}, p)
    with pytest.raises(ValueError, match="not a frozen artifact"):
        load_frozen(str(p))


def test_wrong_config_fails_loudly_rather_than_loading_partially(tmp_path):
    """The failure mode the artifact exists to prevent, checked directly."""
    cfg = build_cfg("sane_ns", cascade=True, reservoirs=5)
    p = save_frozen(_model(**cfg), cfg, str(tmp_path / "c.pt"))
    blob = torch.load(p, map_location="cpu", weights_only=False)
    blob["cfg"] = build_cfg("sane_ns")           # cascade silently dropped
    torch.save(blob, p)
    with pytest.raises(RuntimeError):
        load_frozen(str(p))


def test_build_cfg_leaves_the_baseline_untouched_when_nothing_is_asked():
    from titans.worldmodel.atlas import CONFIGS
    assert build_cfg("sane_ns") == dict(CONFIGS["sane_ns"])


def test_every_cascade_flag_reaches_the_memory():
    """No inert flags. Each one must be observable on the built module.

    This repo has repeatedly shipped flags that were accepted, reported in the
    result JSON, and silently did nothing (`gamma` at c>1, `window_size` in the
    naive path). Centralising the config builder makes that failure *easier* --
    one dropped line now affects three callers at once -- so it is pinned here.
    """
    cfg = build_cfg("sane_ns", cascade=True, reservoirs=7, leak="surprise",
                    leak_rate=0.25, readout="mean", selective=True,
                    gate="content", topology="chain")
    mem = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                     memory_num_layers=2, **cfg).blocks[0].memory
    assert mem.use_cascade_momentum
    assert mem.cascade.num_reservoirs == 7
    assert mem.cascade_leak == "surprise"
    assert mem.cascade_leak_rate == 0.25
    assert mem.cascade.selective
    assert mem.cascade.gate_signal == "content"
    assert mem.cascade.topology == "chain"

    # topology travels too; it is the one that went inert in the 2026-08-06 refactor
    indep = build_cfg("sane_ns", cascade=True, reservoirs=7, topology="independent")
    mem2 = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                      memory_num_layers=2, **indep).blocks[0].memory
    assert mem2.cascade.topology == "independent"


def test_independent_topology_refuses_the_selective_gate():
    """No edges to gate, so the combination is meaningless and must not be accepted."""
    cfg = build_cfg("sane_ns", cascade=True, topology="independent", selective=True)
    with pytest.raises(ValueError, match="no coupling to gate"):
        WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                   memory_num_layers=2, **cfg)


def test_topology_actually_changes_the_computation():
    """`independent` is the null hypothesis (K uncoupled filters); it must differ."""
    z = torch.randn(2, 8, 8)
    outs = []
    for topo in ("chain", "independent"):
        torch.manual_seed(0)
        cfg = build_cfg("sane_ns", cascade=True, reservoirs=4, topology=topo)
        m = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                       memory_num_layers=2, **cfg).eval()
        with torch.no_grad():
            outs.append(m(z))
    assert not torch.allclose(outs[0], outs[1])


def test_no_memory_ablation_removes_the_branch():
    """The denominator for every 'the memory helps' claim in the project."""
    cfg = build_cfg("sane_ns", use_memory=False)
    m = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                   memory_num_layers=2, **cfg)
    assert m.blocks[0].use_memory is False
    z = torch.randn(2, 8, 8)
    assert torch.isfinite(m(z)).all()


def test_memory_hidden_dim_reaches_the_mlp():
    """The capacity axis: it must change the stored weights, not just be accepted."""
    small = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                       memory_num_layers=2, **build_cfg("sane_ns", memory_hidden_dim=16))
    big = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=4,
                     memory_num_layers=2, **build_cfg("sane_ns", memory_hidden_dim=64))
    assert small.blocks[0].memory.memory.params["w0"].shape[0] == 16
    assert big.blocks[0].memory.memory.params["w0"].shape[0] == 64


def test_memory_chunk_can_be_pinned_independently_of_the_segment():
    """Isolating 'wider attention window' from 'longer inner chunk': without this
    the two move together and a seg sweep cannot be interpreted."""
    cfg = build_cfg("sane_ns", memory_chunk_size=8)
    m = WorldModel(latent_dim=8, dim=8, depth=1, segment_size=32,
                   memory_num_layers=2, **cfg)
    assert m.blocks[0].segment_size == 32
    assert m.blocks[0].memory_chunk_size == 8
