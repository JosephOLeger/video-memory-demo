"""Demo interactive texte : Qwen seul vs Qwen + memoire greffee, sur une histoire.

On colle une histoire (ou on en tire une du banc v2 ou v3), une question et jusqu'a 4 options.
Le serveur coupe l'histoire en deux, exactement dans le regime de ``train_text_ab`` :

  * le passe   : tout sauf les ``--window`` derniers jetons, lu en flux par blocs de 256 jetons
                 (regime du checkpoint) et ecrit dans une memoire fraiche ;
  * la fenetre : les ``--window`` derniers jetons, que Qwen lit directement.

Puis il pose la question a trois lecteurs :
  * Qwen seul        : la fenetre seulement (graft coupe) ;
  * Qwen + memoire   : la meme fenetre, plus la memoire qui a ecrit le passe ;
  * Qwen plein texte : toute l'histoire dans le contexte (le plafond, hors flux).

Score des options : vraisemblance de la reponse libre (somme des log p du texte de l'option), le
format entraine, normalisee en probabilites sur les options. Plus une reponse libre (8 jetons).
Banc tire : on peut changer de variante (4 versions qui ne different que par les faits).

Bibliotheque standard seulement. Un seul GPU, un verrou. A lancer dans ~/work/iclr/text (tour) :

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=pylib python demo_text_server.py \\
        --ckpt "Memoire profonde (v3)=runs3/v3_smixA_1500.pt" \\
        --ckpt "Cache 8 etats (v3)=runs3/v3_smixA_kvfull.pt" \\
        --ckpt "Memoire 3,3 M (v2, flux long)=runs_ab/smixA_projres_lencurr.pt" --port 8766
    # en local : ssh -N -L 8766:127.0.0.1:8766 josephleger@172.31.4.119   puis http://localhost:8766
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pylib"))
import gen_story2 as G2  # noqa: E402
import gen_story3 as G3  # noqa: E402
from titans.graft.branch import MemoryBranch, memory_kwargs_from_args  # noqa: E402
from titans.graft.qwen import disabled, graft  # noqa: E402
from train_text_ab import (END, SYS, TextHost, answer_logp, pad_batch,  # noqa: E402
                           write_stream)

BENCHES = {"v3": G3, "v2": G2}
INTRO = "You are reading a long story in pieces.\n\n"
DEFAULTS = dict(mem_type="deep", oracle_write=False, proj_hidden=0, proj_residual=False, write_attnpool=0,
                attnpool_heads=2, write_pool=0, train_down=False, slots=0, tau_init=10.0)


class KVFull(nn.Module):
    """Le cache a budget egal de train_text_ab : `slots` etats bruts de la couche 24, lus par attention."""
    def __init__(self, D, dk, n, tau):
        super().__init__()
        self.slots, self.dim, self.dk = n, D, dk
        self.k_proj, self.q_proj, self.v_proj = (nn.Linear(D, dk, bias=False) for _ in range(3))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

    def init_memory(self, B):
        return {"slots": torch.zeros(B, 0, self.dim, device=next(self.parameters()).device)}, {}

    def write(self, z):
        d0 = torch.zeros(z.shape[0], z.shape[1], device=z.device)
        return {"slots": z[:, -self.slots:]}, {}, (d0, d0, d0)

    def read(self, state, h):
        s_ = state["slots"]
        if s_.shape[1] == 0:
            return torch.zeros(*h.shape[:-1], self.dk, device=h.device, dtype=h.dtype)
        l2 = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-6)  # noqa: E731
        att = torch.softmax(self.log_tau.exp() * torch.einsum("bnd,bkd->bnk", l2(self.q_proj(h)),
                                                              l2(self.k_proj(s_))), -1)
        return torch.einsum("bnk,bkd->bnd", att, self.v_proj(s_))


class ResMLP(nn.Module):
    def __init__(self, lin, din, H, dout):
        super().__init__()
        self.lin = lin
        self.mlp = nn.Sequential(nn.Linear(din, H), nn.GELU(), nn.Linear(H, dout, bias=False))

    def forward(self, x):
        return self.lin(x) + self.mlp(x)


class PostRes(nn.Module):
    def __init__(self, lin, d, H):
        super().__init__()
        self.lin = lin
        self.mlp = nn.Sequential(nn.Linear(d, H), nn.GELU(), nn.Linear(H, d, bias=False))

    def forward(self, x):
        y = self.lin(x)
        return y + self.mlp(y)


class Mem:
    """Une memoire entrainee, reconstruite depuis son checkpoint (memes chemins que train_text_ab)."""
    def __init__(self, label, path, device, stream=(256, "slide")):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        t = types.SimpleNamespace(**{**DEFAULTS, **blob["args"]})       # anciens checkpoints : drapeaux absents
        self.label, self.path, self.t, self.step = label, path, t, blob.get("step")
        self.kind = t.mem_type
        # anciens checkpoints sans regime de flux : on lit le passe en flux (regime de la demo)
        trained = getattr(t, "past_chunk", 0)
        self.past_chunk = trained or stream[0]
        self.past_mode = getattr(t, "past_mode", "chunk") if trained else stream[1]
        self.stream_note = "" if trained else " (entraînée en plein contexte)"
        if self.kind not in ("deep", "kvfull"):
            raise ValueError(f"{path}: mem_type {self.kind} non pris en charge par la demo")
        if t.oracle_write:
            raise ValueError(f"{path}: checkpoint d'oracle (ecrit les faits connus), pas une ecriture apprise")
        bpath = t.basis.format(k=t.d_mem)
        if not os.path.isabs(bpath) and not os.path.exists(bpath):
            bpath = os.path.join(HERE, os.path.basename(bpath))
        pca = torch.load(bpath, map_location="cpu", weights_only=False)["pca"]
        b = MemoryBranch(2048, t.d_mem, chunk_size=t.chunk, write_stride=t.write_stride, down_pca=pca,
                         **memory_kwargs_from_args(blob["args"]))
        if self.kind == "kvfull":
            t.slots = t.slots or max(1, (2 * t.memory_hidden * t.d_mem + 2047) // 2048)
            b.memory = KVFull(2048, t.d_mem, t.slots, t.tau_init)
            b.slot_q = nn.Parameter(torch.zeros(t.slots, t.d_mem))
            b.slot_k = nn.Linear(2048, t.d_mem, bias=False)

            def _delta(self, h, state=None, S=None, *, token_offset=0, write_mask=None):
                if state is None:
                    state, S = self.memory.init_memory(h.shape[0])
                r = self.memory.read(state, h)
                if self.norm_r is not None:
                    r = self.norm_r(r)
                return self.up(r), state, S
            b.delta = types.MethodType(_delta, b)
        if t.proj_hidden and not t.proj_residual:
            H, dm = t.proj_hidden, t.d_mem
            for name in ("k_proj", "v_proj", "q_proj"):
                setattr(b.memory, name, nn.Sequential(nn.Linear(dm, H), nn.GELU(), nn.Linear(H, dm)))
            b.up = nn.Sequential(nn.Linear(dm, H), nn.GELU(), nn.Linear(H, 2048, bias=False))
        if t.write_attnpool > 1:
            b.attn_pool = nn.Linear(t.d_mem, t.attnpool_heads)
        if t.proj_hidden and t.proj_residual:
            H, dm = t.proj_hidden, t.d_mem
            for name in ("k_proj", "v_proj", "q_proj"):
                lin = getattr(b.memory, name)
                setattr(b.memory, name, PostRes(lin, dm, H) if self.kind == "kvfull" else ResMLP(lin, dm, H, dm))
            b.up = ResMLP(b.up, dm, H, 2048)
        res = b.load_state_dict(blob["branch"], strict=False)
        if res.missing_keys or res.unexpected_keys:
            raise ValueError(f"{path}: architecture reconstruite differente du checkpoint ({res})")
        self.branch = b.to(device).eval()
        st0, _ = (b.memory.init_memory(1) if self.kind == "kvfull" else b.init_carry(1))
        self.state_floats = (t.slots * 2048 if self.kind == "kvfull"
                             else sum(v.numel() for v in st0.values()))
        learned_down = bool(t.train_down or t.proj_residual or (t.proj_hidden and not t.proj_residual))
        self.params = sum(p.numel() for n_, p in b.named_parameters()
                          if learned_down or not n_.startswith("down."))

    @torch.no_grad()
    def write(self, h):
        """h [1, N, 2048] (etats couche 24 du passe) -> (state, S, nombre d'ecritures). Ecriture apprise."""
        b, t = self.branch, self.t
        if self.kind == "kvfull":
            kk = b.slot_k(h)
            kk = kk / (kk.norm(dim=-1, keepdim=True) + 1e-6)
            qn = b.slot_q / (b.slot_q.norm(dim=-1, keepdim=True) + 1e-6)
            w_ = torch.softmax(10.0 * torch.einsum("kd,bnd->bkn", qn, kk), -1)
            st, S, _ = b.memory.write(torch.einsum("bkn,bnd->bkd", w_, h))
            return st, S, t.slots
        zz = b.down(h)
        if t.write_attnpool > 1:
            k_ = t.write_attnpool
            n = zz.shape[1] // k_ * k_
            zz = zz[:, :n].view(1, -1, k_, zz.shape[-1])
            w_ = torch.softmax(b.attn_pool(zz), dim=2)
            z = (w_.unsqueeze(-1) * zz.unsqueeze(3)).sum(2).mean(2)
        elif t.write_pool > 1:
            n = zz.shape[1] // t.write_pool * t.write_pool
            z = zz[:, :n].view(1, -1, t.write_pool, zz.shape[-1]).mean(2)
        else:
            z = zz[:, ::t.write_stride]
        st, S = write_stream(b, z, t.chunk)[:2]
        return st, S, z.shape[1]

    def meta(self):
        t = self.t
        return {"label": self.label, "file": os.path.basename(self.path), "step": self.step,
                "kind": "cache de 8 états bruts" if self.kind == "kvfull" else "mémoire profonde (dans des poids)",
                "state_kib": round(self.state_floats * 4 / 1024, 1), "params": self.params,
                "stream": (f"en flux, blocs de {self.past_chunk} jetons" + (" glissants" if self.past_mode == "slide" else "")
                           + self.stream_note) if self.past_chunk else "plein contexte",
                "write": (f"attention apprise par blocs de {t.write_attnpool}" if t.write_attnpool > 1
                          else "8 états choisis par attention" if self.kind == "kvfull"
                          else f"1 jeton sur {t.write_stride}")}


class Demo:
    def __init__(self, a):
        self.a = a
        self.host = TextHost(a.model, a.device)
        self.tok = self.host.tok
        self.mems = []
        for spec in a.ckpt:
            label, _, path = spec.rpartition("=")
            self.mems.append(Mem(label or os.path.basename(path), path, a.device, (a.past_chunk, a.past_mode)))
            print(f"charge : {self.mems[-1].label} <- {path}", flush=True)
        self.wrapper = graft(self.host.model, self.mems[0].branch, layer=self.host.site)
        self.wrapper.writes_enabled = False
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ banc
    def story(self, bench, group, variant, P, D):
        G = BENCHES[bench]
        st = G.make_story(int(group), "eval")
        L = G.layout(st, self.tok, P=int(P), W=self.a.window, D=int(D))[int(variant)]
        segs = []
        for s_, e_, kind in L["bounds"]:
            txt = self.tok.decode(L["past"][s_:e_])
            if kind == "dist":
                kind = "scen"                      # pour le lecteur : un evenement sans importance = du decor
            if segs and segs[-1]["kind"] == kind == "scen":
                segs[-1]["text"] += txt; segs[-1]["b"] = e_
            else:
                segs.append({"text": txt, "kind": kind, "a": s_, "b": e_})
        P_ = len(L["past"])
        segs.append({"text": self.tok.decode(L["window"]), "kind": "window", "a": P_, "b": P_ + len(L["window"])})
        qs = []
        for f in G.FAMILIES:
            it = st["items"][f][int(variant)]
            qs.append({"family": f, "question": it["question"], "options": list(it["options"]),
                       "answer": it["answer"]})
        return {"segments": segs, "questions": qs, "n_tokens": len(L["past"]) + len(L["window"]),
                "bench": bench, "group": int(group), "variant": int(variant), "P": int(P)}

    # ------------------------------------------------------------- inference
    def _prompt(self, ctx_ids, question, notes=None):
        h = self.host
        ids = h.t(SYS + INTRO)
        if notes is not None:
            ids += h.t("Notes kept from earlier parts of the story:\n") + notes + h.t("\n\n")
        ids += h.t("The most recent part of the story:\n") + ctx_ids
        return ids + h.t(f"\n\nQuestion: {question}\nAnswer with a few words." + END)

    def _score(self, q_ids, options):
        if not options:
            return None
        seqs, ql, al = [], [], []
        for o in options:
            at = self.host.t(o[0].upper() + o[1:])
            seqs.append(q_ids + at); ql.append(len(q_ids)); al.append(len(at))
        ids, att = pad_batch(seqs, self.host.pad)
        lp = answer_logp(self.host.model, ids, att, ql, al, self.a.device)
        sc = torch.tensor([float(lp[j].gather(1, torch.tensor(seqs[j][ql[j]:ql[j] + al[j]],
                                                              device=lp[j].device)[:, None]).sum())
                           for j in range(len(seqs))])
        return torch.softmax(sc, 0).tolist()

    def _free(self, q_ids, n=16):
        cur, out = list(q_ids), []
        for _ in range(n):
            lg = self.host.model(input_ids=torch.tensor([cur], device=self.a.device), use_cache=False).logits
            t = int(lg[0, -1].argmax())
            if t in (self.host.im_end, self.tok.eos_token_id):
                break
            out.append(t); cur.append(t)
        txt = self.tok.decode(out).strip()
        return re.split(r"(?<=[.!?])\s", txt)[0]          # la premiere phrase (la generation se repete ensuite)

    @torch.no_grad()
    def run(self, req):
        m = self.mems[int(req.get("mem", 0))]
        text = req["text"]
        question = req["question"].strip()
        options = [o.strip() for o in req.get("options", []) if o.strip()]
        W = self.a.window
        spec = req.get("spec")
        if spec and spec.get("text") == text:
            # histoire du banc non modifiee : les jetons exacts du banc (re-tokeniser le texte decale les blocs)
            L = BENCHES[spec["bench"]].layout(BENCHES[spec["bench"]].make_story(int(spec["group"]), "eval"), self.tok,
                                              P=int(spec["P"]), W=W, D=int(spec.get("D", 300)))[int(spec["variant"])]
            past, win = L["past"], L["window"]
        else:
            ids = self.tok.encode(text if text.startswith(" ") else " " + text, add_special_tokens=False)
            past, win = ids[:-W], ids[-W:]
        ids = past + win
        if len(past) < 16:
            raise ValueError(f"histoire trop courte : {len(ids)} jetons, il en faut plus que la fenetre ({W})")
        if len(ids) > self.a.max_tokens:
            raise ValueError(f"histoire trop longue : {len(ids)} jetons (max {self.a.max_tokens})")
        out = {"n_past": len(past), "n_window": len(win), "window_text": self.tok.decode(win),
               "mem": m.meta(), "options": options}
        with self.lock:
            t0 = time.time()
            self.wrapper.branch = m.branch
            TextHost.past_chunk, TextHost.past_mode = m.past_chunk, m.past_mode
            with disabled(self.wrapper):
                h = self.host.layer_states(torch.tensor([past], device=self.a.device))
            state, S, nw = m.write(h)
            out["writes"], out["write_s"] = nw, round(time.time() - t0, 2)
            q_win = self._prompt(win, question)
            with disabled(self.wrapper):
                out["base"] = {"probs": self._score(q_win, options), "free": self._free(q_win)}
                if req.get("full", True):
                    q_full = self._prompt(ids, question)
                    out["full"] = {"probs": self._score(q_full, options), "free": self._free(q_full)}
            rows = torch.zeros(max(1, len(options)), dtype=torch.long, device=self.a.device)
            rep = lambda d: {k: v.index_select(0, rows) for k, v in d.items()}  # noqa: E731
            self.wrapper.carry, self.wrapper.token_offset = (rep(state), rep(S)), 0
            out["memory"] = {"probs": self._score(q_win, options)}
            self.wrapper.carry, self.wrapper.token_offset = (state, S), 0
            out["memory"]["free"] = self._free(q_win)
            self.wrapper.carry = (None, None)
            out["seconds"] = round(time.time() - t0, 2)
        return out

    def meta(self):
        return {"mems": [m.meta() for m in self.mems], "window": self.a.window, "max_tokens": self.a.max_tokens,
                "benches_shown": self.a.benches.split(","),
                "benches": {"v3": {"families": G3.FAMILIES, "P": [4096, 8192]},
                            "v2": {"families": G2.FAMILIES, "P": [2048, 4096, 8192]}}}


def make_handler(demo: Demo):
    page = os.path.join(HERE, "demo_text.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    with open(page, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                elif u.path == "/meta":
                    self._json(demo.meta())
                else:
                    self._json({"error": "introuvable"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                if path == "/story":
                    self._json(demo.story(req["bench"], req["group"], req["variant"], req["P"], req.get("D", 300)))
                elif path == "/run":
                    self._json(demo.run(req))
                else:
                    self._json({"error": "introuvable"}, 404)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--ckpt", action="append", required=True, help="'Nom affiche=chemin.pt', repetable")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--past-chunk", type=int, default=256, dest="past_chunk",
                    help="flux pour les checkpoints qui n'en ont pas (0 = plein contexte)")
    ap.add_argument("--past-mode", choices=["chunk", "slide"], default="slide", dest="past_mode")
    ap.add_argument("--max-tokens", type=int, default=16384, dest="max_tokens")
    ap.add_argument("--benches", default="v3,v2", help="bancs proposes dans la page, dans l'ordre")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    demo = Demo(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(demo))
    print(f"demo sur http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
