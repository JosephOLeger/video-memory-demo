"""Demo interactive : Qwen seul vs Qwen + memoire greffee sur le banc couleur.

On choisit une video (tenue a l'ecart de l'entrainement), un instant et une couleur.
Le serveur insere l'aiguille (``needle_frames`` images d'aplat) a cet instant dans
le flux echantillonne (``total_frames`` images sur toute la video), puis pose le
QCM couleur aux deux modeles, avec EXACTEMENT le regime de ``train_color`` :

  * Qwen seul        : ne voit que la fenetre recente (les ``window`` dernieres images) ;
  * Qwen + memoire   : la meme fenetre, plus la memoire qui a ecrit tout le passe.

Si l'instant tombe dans la fenetre, les deux modeles voient l'aiguille (le plafond).
Couleur ``aucune`` : pas d'aiguille, la memoire n'a rien a rappeler (controle).

Bibliotheque standard seulement (pas de flask dans l'env). Un seul GPU, un seul
modele : les inferences sont serialisees par un verrou.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=3 python -m titans.graft.demo_server \\
        --ckpt runs/cf_lvb_lr3e3.pt --basis cache/qwen_l24_pca128.pt \\
        --data data/longvideobench --port 8765
    # en local : ssh -N -L 8765:127.0.0.1:8765 josephleger@172.31.4.135
    # puis http://localhost:8765
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

from .branch import MemoryBranch, memory_kwargs_from_args
from .needle import COLORS, LETTERS, build_prompt, make_needle
from .qwen import default_layer, disabled, find_decoder_layers, graft
from .train_color import answer_logits, host_states, video_pool, write_grad

HERE = os.path.dirname(os.path.abspath(__file__))


class Demo:
    def __init__(self, a):
        from transformers import AutoProcessor

        from .check_qwen import load_host

        self.a = a
        blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        t = blob["args"]
        self.targs = t
        self.T, self.W, self.nf = t["total_frames"], t["window"], t["needle_frames"]
        self.max_side, self.chunk, self.stride = t["max_side"], t["chunk"], t["write_stride"]
        # les videos d'eval de l'entrainement (tete de tranche) : jamais vues
        self.vids = video_pool(a.data, t["split"], t["duration_group"],
                               t["eval_videos"], "lvb")
        self.by_name = {v["name"]: v for v in self.vids}

        self.model = load_host(a.model, torch.bfloat16, a.device)
        self.processor = AutoProcessor.from_pretrained(a.model)
        self.site = (t["layer"] if t.get("layer") is not None
                     else default_layer(len(find_decoder_layers(self.model))))
        pca = torch.load(a.basis, map_location="cpu", weights_only=False)["pca"]
        self.branch = MemoryBranch(int(self.model.config.text_config.hidden_size),
                                   t["d_mem"], chunk_size=self.chunk,
                                   write_stride=self.stride, down_pca=pca,
                                   **memory_kwargs_from_args(t)).to(a.device)
        self.branch.load_state_dict(blob["branch"])
        self.branch.eval()
        self.wrapper = graft(self.model, self.branch, layer=self.site)
        self.wrapper.writes_enabled = False
        self.prompt = build_prompt()
        self.letter_ids = torch.tensor(
            [self.processor.tokenizer.encode(LETTERS[i], add_special_tokens=False)[0]
             for i in range(len(COLORS))], device=a.device)
        self.frames, self.thumbs = {}, {}
        self.lock = threading.Lock()
        print(f"pret : {a.ckpt} (pas {blob.get('step')}, eval {blob.get('eval')}), "
              f"{len(self.vids)} videos", flush=True)

    # ------------------------------------------------------------------ donnees
    def sampled(self, name):
        if name not in self.frames:
            from ..worldmodel.data import load_video_frames
            from .fit_basis import resize_frames
            self.frames[name] = resize_frames(
                load_video_frames(self.by_name[name]["path"], num_frames=self.T),
                self.max_side)
        return self.frames[name]

    def thumb_strip(self, name) -> bytes:
        """Bandeau JPEG de 32 vignettes uniformes sur la video (la frise)."""
        if name not in self.thumbs:
            from PIL import Image
            allf = self.sampled(name)
            h = 48
            step = max(1, len(allf) // 32)               # 32 vignettes lisibles
            ims = [Image.fromarray(f.numpy()) for f in allf[::step]]
            w = max(1, round(ims[0].width * h / ims[0].height))
            strip = Image.new("RGB", (w * len(ims), h))
            for i, im in enumerate(ims):
                strip.paste(im.resize((w, h)), (i * w, 0))
            buf = io.BytesIO()
            strip.save(buf, "JPEG", quality=80)
            self.thumbs[name] = buf.getvalue()
        return self.thumbs[name]

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def run(self, name: str, color: int, t_insert: float) -> dict:
        vid = self.by_name[name]
        dur = vid["dur"]
        allf = self.sampled(name)
        fps = self.T / dur
        seq, pos = allf, None
        if color >= 0:
            pos = min(self.T, max(0, round(t_insert / dur * self.T)))
            _, h, w, _ = allf.shape
            seq = torch.cat([allf[:pos], make_needle(self.nf, h, w, COLORS[color][1]),
                             allf[pos:]], 0)
        win, past = seq[-self.W:], seq[:-self.W]
        n_past = past.shape[0]
        if pos is None:
            where = "aucune"
        elif pos + self.nf <= n_past:
            where = "passe"
        elif pos >= n_past:
            where = "fenetre"
        else:
            where = "a cheval"

        def probs(frames):
            lg = answer_logits(self.model, self.processor, frames, self.prompt, fps,
                               self.a.device, self.letter_ids)
            return torch.softmax(lg.float(), -1).tolist()

        with self.lock:
            t0 = time.time()
            with disabled(self.wrapper):
                p_base = probs(win)
                st = host_states(self.model, self.processor, past, fps, self.a.device,
                                 self.site)
            state, S, nw = write_grad(self.branch, st, self.chunk, self.stride)
            self.wrapper.carry, self.wrapper.token_offset = (state, S), 0
            p_mem = probs(win)
            elapsed = time.time() - t0
        after = None
        if where == "passe":
            after = int(round(nw * (n_past - pos - self.nf) / n_past))
        return {"base": p_base, "memory": p_mem, "gold": color, "where": where,
                "writes": nw, "writes_after": after, "seconds": round(elapsed, 2),
                "window_start_s": dur * (self.T - self.W) / self.T}

    def meta(self) -> dict:
        return {"videos": [{"name": v["name"], "dur": v["dur"]} for v in self.vids],
                "colors": [{"name": n, "rgb": list(c)} for n, c in COLORS],
                "letters": LETTERS[:len(COLORS)], "prompt": self.prompt,
                "total_frames": self.T, "window": self.W,
                "needle_frames": self.nf, "write_stride": self.stride,
                "ckpt": os.path.basename(self.a.ckpt)}


def make_handler(demo: Demo):
    page = os.path.join(HERE, "demo.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _video(self, name):
            """mp4 avec support des requetes Range (necessaire pour se deplacer)."""
            if name not in demo.by_name:
                return self._json({"error": "video inconnue"}, 404)
            path = demo.by_name[name]["path"]
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                elif not m.group(1):
                    start = 0
            end = min(end, start + 4 * 2**20 - 1)       # morceaux de 4 Mo
            with open(path, "rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            self._send(206 if m else 200, body, "video/mp4",
                       {"Accept-Ranges": "bytes",
                        "Content-Range": f"bytes {start}-{end}/{size}"})

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    with open(page, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                elif u.path == "/meta":
                    self._json(demo.meta())
                elif u.path == "/video":
                    self._video(q.get("name", ""))
                elif u.path == "/thumbs":
                    self._send(200, demo.thumb_strip(q["name"]), "image/jpeg")
                else:
                    self._json({"error": "introuvable"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            if urlparse(self.path).path != "/run":
                return self._json({"error": "introuvable"}, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                self._json(demo.run(req["video"], int(req["color"]),
                                    float(req.get("t", 0.0))))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    demo = Demo(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(demo))
    print(f"demo sur http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
