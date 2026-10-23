import sys, torch
a, b = torch.load(sys.argv[1]), torch.load(sys.argv[2])
def eq(x, y):
    if x is None or y is None: return x is y
    return torch.equal(x, y)
ok = eq(a["r"], b["r"]) and all(eq(a["st"][k], b["st"][k]) for k in a["st"]) and \
     all(eq(a["S"][k], b["S"][k]) for k in a["S"]) and set(a["g"]) == set(b["g"]) and \
     all(eq(a["g"][k], b["g"][k]) for k in a["g"])
print("BIT-IDENTICAL" if ok else "DIFFERENT", sys.argv[1].split("/")[-1])
