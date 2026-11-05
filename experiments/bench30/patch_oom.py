import sys
p = sys.argv[1]
s = open(p).read()
old = """        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"  OOM at step {step}", flush=True)
            continue"""
new = """        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            # 03/11: drop the failed step's graph before the next write, else the old graph stays alive while the
            # new one is built and the OOM cascades (m512s: 1324 OOM steps of 2000)
            loss = state = S = l_ce = l_kl = ce_own = ce_oth = aux = None
            keyreg["z"].clear(); keyreg["q"] = None
            torch.cuda.empty_cache()
            print(f"  OOM at step {step}", flush=True)
            continue"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched", p)
