"""gen_rich.py (18/11, decision Josep): bank of RICH filler passages written by Qwen2.5-VL-7B-Instruct (text only),
in the seven surface forms of bench30, about four fixed people (GEN names, later substituted by the story's names).
The passages must not state the kinds of facts bench30 asks about (where people / objects are, colours, codes,
appointment days and times, counts, who gave what, preferences); a sentence filter removes what slips through.
    CUDA_VISIBLE_DEVICES=c python gen_rich.py --out bank_c.jsonl --n 1000 --seed c
Output: jsonl {form, text, n_keep, n_drop, theme}.
"""
import argparse
import json
import random
import re
import time

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

MID = "Qwen/Qwen2.5-VL-7B-Instruct"
GEN = ["Marta", "Jonas", "Priya", "Felix"]
FORMS = ["narration", "diary", "chat", "email", "dialogue", "note", "carelog"]
STYLE = {
    "narration": "a third-person narration, in plain past tense sentences, like a short slice-of-life story",
    "diary": "a first-person diary entry; sentences may start with 'Dear diary,', 'Today', 'In the evening,' and such",
    "chat": "a group chat log; every line looks like '[14:05] Marta: message.' (hour between 07 and 21, short messages)",
    "email": "a thread of short emails; every item looks like 'Email from Marta to Jonas: \"...\"' or 'Jonas wrote in an email: \"...\"'",
    "dialogue": "a spoken conversation written as quotes, like '\"...,\" Marta said.' or '\"...,\" Jonas told Priya.'",
    "note": "a list of short personal notes; every item starts with 'Note:', 'Reminder:', 'Note to self:' or 'Remember:'",
    "carelog": "a home care log kept by a carer; every item starts with 'Care log 09:30:' or 'Carer's note:' and describes the day of the person cared for",
}
THEMES = ["the weather and the garden", "a long phone call with family", "cooking a meal together", "a television programme",
          "a walk in the neighbourhood", "a visit from an old friend", "aches, sleep and feeling tired", "memories of younger years",
          "plans for the weekend", "the neighbours and their dog", "a short trip by train", "the news and an opinion about it",
          "a hobby (knitting, chess, painting, crosswords)", "a pet that misbehaves", "a small repair at home", "a birthday coming up",
          "a book someone is reading", "music and an old record", "a quarrel and a reconciliation", "shopping for groceries",
          "a doctor's general advice about resting", "the church choir", "grandchildren visiting", "a broken appliance",
          "a recipe that went wrong", "a letter from an old colleague", "the bus being late", "photographs in an album"]
AVOID = ("Do not say where any person went, is or was. Do not say where any object was put or is kept. Do not mention "
         "colours of things. Do not write any number, code, price, date, day of the week or clock time inside the sentences "
         "(timestamps at the start of chat or care log lines are fine). Do not say who gave or lent something to whom. "
         "Do not state anyone's favourite thing or preference. Do not count how many times something happened.")

COL = r"\b(red|blue|green|yellow|black|white|brown|purple|orange|grey|gray|pink)\b"
DAYS = r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
ROOMS = ["kitchen", "bathroom", "bedroom", "hallway", "garden", "garage", "living room", "cellar", "attic", "balcony", "study",
         "porch", "pharmacy", "bakery", "park", "library", "post office", "market", "station", "church", "clinic", "bank",
         "museum", "hairdresser", "supermarket", "cinema", "office", "school", "hospital", "shop", "cafe", "café", "dentist"]
PLACE = r"\b(" + "|".join(re.escape(x) for x in ROOMS) + r")\b"
_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z])|(?<=[.!?]["\')\]])\s+(?=["\'(\[]?[A-Z])')


def split_sents(text):
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def keep(s):
    body = re.sub(r"^\[\d\d:\d\d\]\s*", "", s)          # chat timestamp allowed
    body = re.sub(r"^Care log \d\d:\d\d:\s*", "", body)  # care log timestamp allowed
    if re.search(r"\d", body):
        return False
    if re.search(COL, body, re.I) or re.search(DAYS, body) or re.search(PLACE, body, re.I):
        return False
    if re.search(r"\b(am|pm)\b", body):
        return False
    return 3 <= len(body.split()) <= 60


def prompt(form, theme, r):
    names = GEN[:]
    r.shuffle(names)
    who = f"{names[0]}, {names[1]}, {names[2]} and {names[3]}"
    return (f"Write {STYLE[form]}, about everyday life, on the theme: {theme}. The people are {who}; use these four "
            f"first names, and no other names. Length: 180 to 260 words, in one block, no title, no headings. {AVOID} "
            f"Output only the text.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=440, dest="max_new")
    a = ap.parse_args()
    r = random.Random(1811 + a.seed)
    tok = AutoTokenizer.from_pretrained(MID)
    tok.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MID, dtype=torch.bfloat16).to("cuda").eval()
    jobs = [(FORMS[i % len(FORMS)], r.choice(THEMES)) for i in range(a.n)]
    r.shuffle(jobs)
    t0, kept, drop_tot, keep_tot = time.time(), 0, 0, 0
    with open(a.out, "w") as fo:
        for b in range(0, len(jobs), a.batch):
            chunk = jobs[b:b + a.batch]
            txt = [tok.apply_chat_template([{"role": "user", "content": prompt(f, t, r)}], add_generation_prompt=True,
                                           tokenize=False) for f, t in chunk]
            x = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                go = model.generate(**x, max_new_tokens=a.max_new, do_sample=True, temperature=0.9, top_p=0.95)
            outs = [tok.decode(g[x["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in go]
            for (f, t), o in zip(chunk, outs):
                sents = split_sents(o.replace("\n", " "))
                ok = [s for s in sents if keep(s)]
                drop_tot += len(sents) - len(ok)
                keep_tot += len(ok)
                if sum(len(s.split()) for s in ok) >= 100:
                    fo.write(json.dumps(dict(form=f, theme=t, text=" ".join(ok), n_keep=len(ok), n_drop=len(sents) - len(ok))) + "\n")
                    kept += 1
            fo.flush()
            print(f"{b + len(chunk)}/{len(jobs)} kept {kept} sent_keep {keep_tot} sent_drop {drop_tot} "
                  f"{(time.time() - t0) / (b + len(chunk)):.1f} s/passage", flush=True)
    print(f"GEN-RICH-FINIE {a.out} kept {kept}/{a.n}", flush=True)


if __name__ == "__main__":
    main()
