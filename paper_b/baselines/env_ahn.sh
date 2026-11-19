# Source this before ahn_eval.py. AHN needs its own environment (install_ahn_env.sh): torch 2.5.1, transformers 4.51,
# flash-linear-attention, flash-attn, mamba-ssm. The text helpers of paper_b/text are on the path for the prompt and the judge.
B=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH=${AHN_ENV:-$HOME/miniforge3/envs/ahn}/bin:$PATH
export CC=${AHN_ENV:-$HOME/miniforge3/envs/ahn}/bin/gcc
export TRITON_CACHE_DIR=$B/cache/triton TORCHINDUCTOR_CACHE_DIR=$B/cache/inductor TMPDIR=$B/cache   # /tmp may be noexec
export HF_HUB_CACHE=$B/hf HF_HUB_OFFLINE=1
export PYTHONPATH=$B/../text TEXT_GEN=gen_bench30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
