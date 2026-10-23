import json, sys, random, math
from collections import Counter
rows = [json.loads(l) for l in open(sys.argv[1])]
C = [c for c in ['full', 'blind', 'win512', 'win1024', 'other', 'loc1', 'loc3'] if f'{c}_o0' in rows[0]['res']]
am = lambda v: max(range(4), key=lambda i: v[i])
def pred(r, c, o):
    if o == 'avg':
        v = [r['res'][f'{c}_o0'][i] + r['res'][f'{c}_o1'][i] for i in range(4)]
        return am(v)
    return am(r['res'][f'{c}_{o}'])
def ok(r, c, o='avg'): return pred(r, c, o) == r['gold']
def acc(rs, c, o='avg'): return sum(ok(r, c, o) for r in rs) / max(len(rs), 1)
def ci(p, n): return 1.96 * math.sqrt(p * (1 - p) / max(n, 1))
def line(name, rs, o='avg'):
    s = f'| {name} | {len(rs)} |'
    for c in C: s += f' {100*acc(rs,c,o):.1f} |'
    return s
hdr = '| subset | n | ' + ' | '.join(C) + ' |\n|' + '---|' * (len(C) + 2)
print(f'N={len(rows)} articles={len(set(r["aid"] for r in rows))}\n')
print('## accuracy (%), debiased = sum of logprobs over the two option orders\n' + hdr)
print(line('all', rows))
print(line('EASY', [r for r in rows if not r['hard']]))
print(line('HARD', [r for r in rows if r['hard']]))
nt = sorted(r['ntok'] for r in rows); t1, t2 = nt[len(nt)//3], nt[2*len(nt)//3]
print(line(f'len <{t1}', [r for r in rows if r['ntok'] < t1]))
print(line(f'len {t1}-{t2}', [r for r in rows if t1 <= r['ntok'] < t2]))
print(line(f'len >={t2}', [r for r in rows if r['ntok'] >= t2]))
print(f'\n95% half-width at n={len(rows)}, p=0.5: {100*ci(.5,len(rows)):.1f} pts')
print('\n## per order (letter bias)\n' + hdr)
print(line('order o0', rows, 'o0')); print(line('order o1', rows, 'o1'))
print('\npredicted letter distribution (slot) o0 / o1, gold slot dist')
for c in C:
    for o, order in [('o0', [0,1,2,3]), ('o1', [2,3,0,1])]:
        slot = Counter('ABCD'[order.index(am(r['res'][f'{c}_{o}']))] for r in rows)
        print(f'  {c:8s} {o}: ' + ' '.join(f'{k}:{100*slot[k]/len(rows):.0f}' for k in 'ABCD'))
gs = Counter('ABCD'[r['gold']] for r in rows); print('  gold o0: ' + ' '.join(f'{k}:{100*gs[k]/len(rows):.0f}' for k in 'ABCD'))

# useful filter: full right, blind wrong, win512 wrong, win1024 wrong
def useful(r, o): return ok(r, 'full', o) and not ok(r, 'blind', o) and not ok(r, 'win512', o) and not ok(r, 'win1024', o)
print('\n## filter "useful" (full ok, blind wrong, win512 wrong, win1024 wrong)')
for o in ['avg']:
    U = [r for r in rows if useful(r, o)]
    print(f'in-sample (select and score on debiased avg): n={len(U)} ({100*len(U)/len(rows):.1f}%)')
    print(hdr); print(line('useful (in-sample)', U))
# cross-fit: select on order o0, measure on order o1 and vice versa
for so, mo in [('o0', 'o1'), ('o1', 'o0')]:
    U = [r for r in rows if useful(r, so)]
    print(f'\ncross-fit: select on {so}, score on {mo}: n={len(U)}')
    print(hdr); print(line(f'useful sel {so} -> {mo}', U, mo))
# split-half by article: selection rule on half A can only transfer through observable features;
# report subset size stability and the cross-fit on each half
aids = sorted(set(r['aid'] for r in rows)); random.Random(1).shuffle(aids); HA = set(aids[:len(aids)//2])
for nm, part in [('half A', [r for r in rows if r['aid'] in HA]), ('half B', [r for r in rows if r['aid'] not in HA])]:
    U = [r for r in part if useful(r, 'o0')]
    print(f'\n{nm}: n={len(part)}, selected on o0: {len(U)} ({100*len(U)/len(part):.1f}%); scored on o1:')
    print(hdr); print(line(nm, U, 'o1'))
# feature rule learned on half A (best margin full - max(blind,win1024) among HARD x len x locality cells), applied to B
def cell(r): return (r['hard'], 'long' if r['ntok'] >= t2 else 'short', 'loc' if ok(r, 'loc3', 'o0') else 'glob')
def margin(rs, o): return acc(rs, 'full', o) - max(acc(rs, 'blind', o), acc(rs, 'win1024', o))
A = [r for r in rows if r['aid'] in HA]; B = [r for r in rows if r['aid'] not in HA]
cells = sorted(set(cell(r) for r in rows))
print('\n## cells (hard, length, loc3 ok on o0): margin full - max(blind, win1024); A=select half, B=held-out half, scored on o1')
print('| cell | nA | margin A | nB | margin B |\n|---|---|---|---|---|')
for c in cells:
    a = [r for r in A if cell(r) == c]; b = [r for r in B if cell(r) == c]
    print(f'| {c} | {len(a)} | {100*margin(a,"o1"):.1f} | {len(b)} | {100*margin(b,"o1"):.1f} |')

# local vs global labelling
if 'loc1' in C:
    print('\n## local/detail vs global/gist (label: loc1 or loc3 correct on debiased avg)')
    for lab, f in [('local1 (top1 sentence suffices)', lambda r: ok(r, 'loc1')),
                   ('local3 (top3 sentences suffice, not top1)', lambda r: ok(r, 'loc3') and not ok(r, 'loc1')),
                   ('global (neither)', lambda r: not ok(r, 'loc1') and not ok(r, 'loc3'))]:
        S = [r for r in rows if f(r)]
        print(f'{lab}: n={len(S)} ({100*len(S)/len(rows):.1f}%)')
    print(hdr)
    for lab, f in [('local (loc3 ok)', lambda r: ok(r, 'loc3')), ('global (loc3 wrong)', lambda r: not ok(r, 'loc3')),
                   ('local, blind wrong', lambda r: ok(r, 'loc3') and not ok(r, 'blind')),
                   ('global, blind wrong', lambda r: not ok(r, 'loc3') and not ok(r, 'blind'))]:
        print(line(lab, [r for r in rows if f(r)]))
    # cross-fit labels: label on o0, score on o1
    print('\ncross-fit (label on o0, score on o1)\n' + hdr)
    print(line('local (loc3 ok o0)', [r for r in rows if ok(r, 'loc3', 'o0')], 'o1'))
    print(line('global (loc3 wrong o0)', [r for r in rows if not ok(r, 'loc3', 'o0')], 'o1'))
    print(line('global&blind wrong o0', [r for r in rows if not ok(r, 'loc3', 'o0') and not ok(r, 'blind', 'o0')], 'o1'))
    print('\ncross-fit among blind-wrong (o0): local = loc3 ok on o0, scored on o1\n' + hdr)
    BW = [r for r in rows if not ok(r, 'blind', 'o0')]
    print(line('blind-wrong, local', [r for r in BW if ok(r, 'loc3', 'o0')], 'o1'))
    print(line('blind-wrong, global', [r for r in BW if not ok(r, 'loc3', 'o0')], 'o1'))
    print(line('blind-wrong, HARD, global', [r for r in BW if r['hard'] and not ok(r, 'loc3', 'o0')], 'o1'))
    U = [r for r in rows if useful(r, 'o0')]
    print(f'\nuseful (sel o0) split: local {sum(ok(r,"loc3","o0") for r in U)} / global {sum(not ok(r,"loc3","o0") for r in U)}')
