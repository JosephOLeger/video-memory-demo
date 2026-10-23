"""Copie un checkpoint de distillation en ajoutant val_mem (eval_lvb l'affiche)."""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
b = torch.load(src, map_location="cpu", weights_only=False)
b.setdefault("val_mem", float("nan"))
torch.save(b, dst)
print(dst, b["step"], sorted(b["args"])[:3], b["args"].get("write_stride"), b["args"].get("chunk"))
