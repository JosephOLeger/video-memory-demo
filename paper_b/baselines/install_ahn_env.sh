# One-off: the AHN environment (conda env named ahn, python 3.11). The official repository goes in src_dl/AHN-main
# (github.com/ByteDance-Seed/AHN). Two fixes applied by hand after this, see README.md.
set -ex
E=${AHN_ENV:-$HOME/miniforge3/envs/ahn}
export PATH=$E/bin:$PATH
PIP="$E/bin/python -m pip"
$PIP install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$PIP install transformers==4.51.0 accelerate safetensors einops sentencepiece numpy==1.26.4 "huggingface_hub<1.0" ninja packaging
$PIP install "git+https://github.com/Seerkfang/flash-linear-attention.git@main#egg=flash-linear-attention"
$PIP install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
MAMBA_SKIP_CUDA_BUILD=TRUE $PIP install --no-build-isolation --no-deps "git+https://github.com/yuweihao/mamba.git"
cd "$(dirname "$0")/src_dl/AHN-main" && $PIP install --no-deps -e .
