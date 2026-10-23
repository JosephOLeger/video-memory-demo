import glob, sys
import torch
nT = nW = n = 0
sT = sW = sn = 0
for f in glob.glob(sys.argv[1] + "/*.pt"):
    d = torch.load(f, weights_only=False)
    for q in d["qs"]:
        okT = int(q["lt_teacher"].argmax()) == q["answer"]
        okW = int(q["lt_window"].argmax()) == q["answer"]
        if q["kind"] == "shape":
            sn += 1; sT += okT; sW += okW
            continue
        n += 1; nT += okT; nW += okW
        print(" ", q["c"], q["p"], "|", q["meta"]["activity"][:50], "->", q["options"][q["answer"]][:25],
              "| opts:", [o[:18] for o in q["options"]], "T", okT, "W", okW)
print("natural n", n, "teacher", nT, "window", nW, "| shape n", sn, "teacher", sT, "window", sW)
