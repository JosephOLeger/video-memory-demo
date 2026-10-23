"""LongVideoBench pour les checkpoints de DISTILLATION : la memoire est ecrite comme a
l'entrainement (clips de --clip images encodes SEULS, tokens video seulement, pas
d'ecriture du checkpoint), et non par un seul passage de Qwen sur les 256 images.

Tout le reste (reference, controles --memory-source, notation par logit de lettre)
est celui de ``titans.graft.eval_lvb`` : on remplace seulement ``write_memory``.

    PYTHONPATH=src python ~/work/iclr/lvb/eval_lvb_clips.py --clip 16 --data ... --ckpt ...
"""
import sys

import torch

import titans.graft.eval_lvb as E
from titans.graft.train_load import video_z

CLIP = 16
if "--clip" in sys.argv:
    i = sys.argv.index("--clip")
    CLIP = int(sys.argv[i + 1])
    del sys.argv[i:i + 2]


@torch.no_grad()
def write_memory_clips(branch, model, processor, frames, prompt, fps, device, layer,
                       chunk, stride):
    zs = [video_z(model, processor, frames[c:c + CLIP], fps, layer, branch.down,
                  device, stride) for c in range(0, frames.shape[0] - CLIP + 1, CLIP)]
    z = torch.cat(zs, 0).float().to(device)[None]
    state, S = branch.init_carry(1)
    state, S = branch.memory.update(z, state, S, min(chunk, z.shape[1]))
    return state, S, int(z.shape[1])


E.write_memory = write_memory_clips
print(f"ecriture par clips de {CLIP} images (comme la distillation)", flush=True)
E.main()
