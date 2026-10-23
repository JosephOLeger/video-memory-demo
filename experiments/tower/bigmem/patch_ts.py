p = "train_stream.py"
s = open(p).read()
a = "        st, S, _, _ = write_stream(branch, z, a.chunk)"
assert s.count(a) == 1
s = s.replace(a, "        st, S, _, _ = (write_stream_ckpt(branch, z, a.chunk, a.ckpt_seg) if a.ckpt_seg\n"
                 "                       else write_stream(branch, z, a.chunk))")
imp = "from .train_load import pr, write_stream"
assert s.count(imp) == 1
s = s.replace(imp, imp + "\nfrom .write_ckpt import write_stream_ckpt")
b = '    ap.add_argument("--max-past", type=int, default=40, dest="max_past", help="clips de memoire au plus")'
assert s.count(b) == 1
s = s.replace(b, b + '\n    ap.add_argument("--ckpt-seg", type=int, default=0, dest="ckpt_seg",\n'
                     '                    help="recalcul par segments de N tokens (0 = non ; gradient identique)")')
open(p, "w").write(s)
print("patch ok")
