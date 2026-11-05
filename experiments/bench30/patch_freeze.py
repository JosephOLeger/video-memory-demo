import sys
p = sys.argv[1]
s = open(p).read()
o1 = '    ap.add_argument("--oracle-mix-steps", type=int, default=0, dest="oracle_mix_steps",\n'
n1 = ('    ap.add_argument("--freeze", default="", help="comma list of branch parameter-name prefixes to freeze "\n'
      '                    "(03/11 two-stage: read = q_proj,norm_r,up,memory.memory.params ; "\n'
      '                    "write = attn_pool,memory.k_proj,memory.v_proj,memory.alpha_head,memory.eta_head,memory.theta_head)")\n' + o1)
o2 = "    params = list(branch.trainable_parameters())\n"
n2 = ("    if a.freeze:\n"
      "        pre_ = [x for x in a.freeze.split(\",\") if x]\n"
      "        frz_ = [n_ for n_, p_ in branch.named_parameters() if any(n_.startswith(x) for x in pre_)]\n"
      "        for n_, p_ in branch.named_parameters():\n"
      "            if n_ in frz_:\n"
      "                p_.requires_grad_(False)\n"
      "        print(f\"freeze: {len(frz_)} tensors frozen: {frz_}\", flush=True)\n" + o2)
assert s.count(o1) == 1 and s.count(o2) == 1
open(p, "w").write(s.replace(o1, n1).replace(o2, n2))
print("patched")
