"""21/10 : train_stream lit les cibles CG-Bench (kind "cgb") : pas d'etats de l'enseignant
(perte sur le sens des reponses), filtre aveugle, recalcul par segments AUSSI a l'entrainement,
pas de passe Qwen inutile (base_states) quand la perte latente est coupee, bilan par distance."""
p = "train_stream.py"
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


# load : drapeau aveugle
rep('                     window_ok=int(q["lt_window"].argmax()) == q["answer"])',
    '                     window_ok=int(q["lt_window"].argmax()) == q["answer"],\n'
    '                     blind_ok=("lt_blind" in q and int(q["lt_blind"].argmax()) == q["answer"]))')
# evaluate : la perte latente n'existe que si l'enseignant a des etats
rep('                       else write_stream(branch, z, a.chunk))\n'
    '        hb = base_states(model, processor, wrapper, vids, q, lid, a.device)\n',
    '                       else write_stream(branch, z, a.chunk))\n'
    '        lat = q.get("h_teacher") is not None\n'
    '        hb = base_states(model, processor, wrapper, vids, q, lid, a.device) if lat else None\n')
rep('                     "lat_mem": float(lat_loss(hm, q, hb, a.device)),\n'
    '                     "lat_wrong": float(lat_loss(hw, q, hb, a.device))})',
    '                     "lat_mem": float(lat_loss(hm, q, hb, a.device)) if lat else 1.0,\n'
    '                     "lat_wrong": float(lat_loss(hw, q, hb, a.device)) if lat else 1.0})')
rep('    for k in ("shape", "natural", "color"):', '    for k in ("shape", "natural", "color", "cgb"):')
rep('    for lo, hi in ((2, 4), (4, 10), (10, 100)):',
    '    for lo, hi in ((2, 5), (5, 15), (15, 100)):                  # cgb : distance en clips\n'
    '        out[f"cgb_d{lo}-{hi}"] = summ([r for r in rows if r["kind"] == "cgb" and lo <= r["dist"] < hi])\n'
    '    for lo, hi in ((2, 4), (4, 10), (10, 100)):')
# entrainement : recalcul par segments, pas de base_states si la perte latente est coupee
rep('                hb = base_states(model, processor, wrapper, vids, q, lid, a.device)\n'
    '                st, S, _, g3 = write_stream(branch, past(vids, q, a.max_past, a.device), a.chunk)\n',
    '                use_lat = a.lam_lat > 0 and q.get("h_teacher") is not None\n'
    '                hb = base_states(model, processor, wrapper, vids, q, lid, a.device) if use_lat else None\n'
    '                zq = past(vids, q, a.max_past, a.device)\n'
    '                st, S, _, g3 = (write_stream_ckpt(branch, zq, a.chunk, a.ckpt_seg) if a.ckpt_seg\n'
    '                                else write_stream(branch, zq, a.chunk))\n')
rep('                lat = h.new_zeros(()) if q.get("variant") else lat_loss(h, q, hb, a.device)',
    '                lat = (lat_loss(h, q, hb, a.device) if use_lat and not q.get("variant")\n'
    '                       else h.new_zeros(()))')
# entrainement : retirer ce qui se resout sans video
rep('    tr = [q for q in qs if not q["eval"] and (q["teacher_ok"] or not a.teacher_ok_only)]',
    '    tr = [q for q in qs if not q["eval"] and (q["teacher_ok"] or not a.teacher_ok_only)\n'
    '          and not (a.drop_blind and q["blind_ok"])]')
rep('    ap.add_argument("--batch", type=int, default=4)',
    '    ap.add_argument("--drop-blind", action="store_true", dest="drop_blind",\n'
    '                    help="entrainement sans les questions que Qwen resout sans video")\n'
    '    ap.add_argument("--batch", type=int, default=4)')
open(p, "w").write(s)
print("patch2 ok")
