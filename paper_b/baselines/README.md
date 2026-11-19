# AHN under our controls

AHN (ByteDance Seed, arXiv 2510.07318) keeps Qwen2.5-3B's attention on a sliding window and writes every token that
leaves it into a recurrent state per layer. We evaluate the published modules (GDN, DN, Mamba2) with our prompt, our
judge and our content control.

## Setup

`install_ahn_env.sh` builds a separate environment (torch 2.5.1, transformers 4.51, flash-linear-attention,
flash-attn, mamba-ssm). The official repository goes in `src_dl/AHN-main`, the modules and the host in `hf/`.
Two fixes are applied by hand and are not in the official code: the mamba-ssm stride fix (commit a096b91 of the
yuweihao fork) and, in `ahn_eval.py`, the sink index at decoding (their code removes the token at a hard-coded 128
instead of `config.num_attn_sinks`). Source `env_ahn.sh` before any command.

## Conditions (`ahn_eval.py`)

With ctx the context tokens, past = ctx[:-W], window = ctx[-W:], W = 256:

| cond | model | input |
|---|---|---|
| `win` | base Qwen2.5-3B | prefix + window + question |
| `full` | base Qwen2.5-3B | the whole context |
| `mem` | AHN | prefix + ctx + question, sinks = the prefix, window = W + question: the past is in the recurrent state |
| `swa` | AHN, read zeroed | same as mem; isolates the sliding-window channel from the compressed memory |
| `omem` | AHN | prefix + past of the previous record + window and question of this one: the content control |

`omem` is implemented by splicing the input, not by copying a state tensor: AHN is causal and the sinks are the
shared prefix, so the state at the first window token is the one built on the other record (checked numerically by
`verify_omem.py`). Output: `<out>/<tag>_<variant>.jsonl` and `_summary.json` (per condition, per type, McNemar
pairs). `run_full_ahn.sh` runs the thirty types and BABILong 4k for the three variants.

## Sizes

| | AHN (any variant) | ours |
|---|---|---|
| trained parameters | 11.8 to 13.0 M | 0.33 M |
| state carried | 9.4 M floats (36 MiB in fp32) | 16.6k weights (65 KB) |

## Host mismatch

AHN is trained on Qwen2.5-3B text; our graft runs on the language model of Qwen2.5-VL-3B. Same tokenizer; the full
context score differs by 3.5 points on 86 questions, within the standard error.
