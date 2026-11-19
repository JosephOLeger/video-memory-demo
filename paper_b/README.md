# Paper B: a memory branch on a frozen Qwen2.5-VL-3B

Three directories, one branch (`src/titans/graft/branch.py`):

* `text/`: the text track. Data generators, the trainer `train_b30.py`, the scorers, and `queue.sh` with the exact
  recipes of every model in the paper.
* `video/`: the video track. FineVideo fetch and audit, the amputated-past queue (`titans.graft.ctx_targets` and
  `train_ctx`), the CG-Bench tools. The earlier video tracks (colour needle, streaming questions, LongVideoBench,
  MCQ distillation) are the modules `train_color`, `train_stream`, `eval_lvb`, `train_mcq` of the library.
* `baselines/`: AHN (ByteDance) under the same prompt, the same judge and the same content control.

The content control is the same everywhere: a memory written from another text or another video, read by the
same window and the same question. A gain that this control also shows is answer style, not recalled content.
