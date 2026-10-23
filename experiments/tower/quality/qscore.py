"""QuALITY dev scoring with Qwen2.5-VL (text only), letter log-likelihood.
Conditions: full, blind, win512, win1024, other, loc1 (best BM25 sentence), loc3 (top3 sentences).
Two option orders: identity and a fixed derangement-ish permutation (cyclic shift by 2).
Writes one jsonl line per question with per-condition, per-order letter logprobs."""
import argparse, json, math, os, random, re, sys, time
from collections import Counter
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--conds', default='full,blind,win512,win1024,other,loc1,loc3')
ap.add_argument('--early', type=int, default=200)
args = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
if 'VL' in args.model:
    from transformers import Qwen2_5_VLForConditionalGeneration as M
else:
    from transformers import AutoModelForCausalLM as M
model = M.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()

arts = [json.loads(l) for l in open(args.data)]
# one article text per article_id; other-article map is a fixed shuffle over distinct ids
texts = {}
for a in arts:
    texts.setdefault(a['article_id'], a['article'])
ids = sorted(texts)
rng = random.Random(0)
perm_ids = ids[:]
while True:
    rng.shuffle(perm_ids)
    if all(x != y for x, y in zip(ids, perm_ids)):
        break
other_of = dict(zip(ids, perm_ids))

def clean(t):
    return re.sub(r'[ \t]*\n[ \t]*\n\s*', '\n\n', re.sub(r'[ \t]+', ' ', t)).strip()

# ---- BM25 over sentences
WORD = re.compile(r"[a-z0-9']+")
STOP = set('the a an of to and in is was were be been it he she they his her their that this what which who whom why how did does do for on at by with as from or not but had has have i you we our your its into than then there so if about after before when while would could should will can'.split())
def toks(s):
    return [w for w in WORD.findall(s.lower()) if w not in STOP]
def sentences(t):
    t = re.sub(r'\s+', ' ', t)
    parts = re.split(r'(?<=[.!?"])\s+(?=[A-Z"\'])', t)
    out, buf = [], ''
    for p in parts:
        buf = (buf + ' ' + p).strip()
        if len(buf.split()) >= 6:
            out.append(buf); buf = ''
    if buf: out.append(buf)
    return out
def bm25_top(sents, query, k):
    docs = [toks(s) for s in sents]
    N = len(docs); avg = sum(map(len, docs)) / max(N, 1)
    df = Counter(w for d in docs for w in set(d))
    q = toks(query)
    sc = []
    for i, d in enumerate(docs):
        tf = Counter(d); s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (N - df[w] + .5) / (df[w] + .5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (.25 + .75 * len(d) / avg))
        sc.append(s)
    top = sorted(range(N), key=lambda i: -sc[i])[:k]
    return [sents[i] for i in sorted(top)]

LET = ['A', 'B', 'C', 'D']
let_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LET]
assert len(set(let_ids)) == 4
ORDERS = {'o0': [0, 1, 2, 3], 'o1': [2, 3, 0, 1]}  # o1[j] = original option index shown at slot j

def build(text, q, opts, order):
    body = ''
    if text is not None:
        body = 'Read the following text and answer the question.\n\n' + text + '\n\n'
    else:
        body = 'Answer the following question.\n\n'
    body += 'Question: ' + q.strip() + '\n'
    for j, oi in enumerate(order):
        body += f'{LET[j]}. {opts[oi].strip()}\n'
    body += 'Answer with the letter of the correct option only.'
    msgs = [{'role': 'user', 'content': body}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def score(prompt):
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    out = model(input_ids=ids, logits_to_keep=1)
    lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    return lp[let_ids].tolist(), ids.shape[1]

conds = args.conds.split(',')
done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        done.add(json.loads(l)['qid'])
fo = open(args.out, 'a')
n = 0; t0 = time.time()
cache_tok = {}
for a in arts:
    text = clean(a['article'])
    if a['article_id'] not in cache_tok:
        cache_tok[a['article_id']] = tok.encode(text, add_special_tokens=False)
    tids = cache_tok[a['article_id']]
    sents = sentences(text)
    otext = clean(texts[other_of[a['article_id']]])
    for qq in a['questions']:
        if args.limit and n >= args.limit: break
        n += 1
        if qq['question_unique_id'] in done: continue
        q, opts, gold = qq['question'], qq['options'], qq['gold_label'] - 1
        query = q + ' ' + ' '.join(opts)
        ctx = {'full': text, 'blind': None,
               'win512': tok.decode(tids[-512:]), 'win1024': tok.decode(tids[-1024:]),
               'other': otext,
               'loc1': ' '.join(bm25_top(sents, query, 1)),
               'loc3': ' '.join(bm25_top(sents, query, 3))}
        rec = {'qid': qq['question_unique_id'], 'aid': a['article_id'], 'gold': gold,
               'hard': qq['difficult'], 'ntok': len(tids), 'res': {}}
        for c in conds:
            for on, order in ORDERS.items():
                lp, L = score(build(ctx[c], q, opts, order))
                # map back to original option indices
                lpo = [None] * 4
                for j, oi in enumerate(order): lpo[oi] = lp[j]
                rec['res'][f'{c}_{on}'] = lpo
        fo.write(json.dumps(rec) + '\n'); fo.flush()
        if n % 50 == 0:
            print(f'{n} q  {time.time()-t0:.0f}s', flush=True)
        if args.early and n == args.early:
            fo.flush()
            rows = [json.loads(l) for l in open(args.out)]
            def acc(c):
                return sum(max(range(4), key=lambda i: r['res'][c][i]) == r['gold'] for r in rows) / len(rows)
            f, b = acc('full_o0'), acc('blind_o0')
            print(f'EARLY CHECK n={len(rows)} full={f:.3f} blind={b:.3f}', flush=True)
            if f - b < 0.05:
                print('STOP: full ~ blind', flush=True); sys.exit(0)
    if args.limit and n >= args.limit: break
print('DONE', n, f'{time.time()-t0:.0f}s', flush=True)
