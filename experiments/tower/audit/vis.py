import sys; sys.path.insert(0,"snap1")
import gen_scenes as G
spec=[("init_count",7,2),("first_left",11,1),("hat_color",3,0),("sign_number",5,3),("door_state",2,2),("order",4,1)]
for fam,g,v in spec:
    it=G.make_group(fam,g)[v]; f=it["facts"]
    lab=f"{fam} g{g}v{v} ans={it['options'][it['answer']]} n0={f['n0']} left={f['left']} ent={f['entered']} door {f['door0']}->{f['door1']} hat={f['hat']} sign={f['sign']}"
    print(lab); print(it["options"])
    G.contact_sheet([it["episode"]], f"vis_{fam}.png", every=2, thumb=160, labels=[lab], ncol=8)
