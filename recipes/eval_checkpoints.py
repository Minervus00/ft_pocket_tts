"""Evaluate and compare Pocket TTS checkpoints for a new language.

Standalone: run it on exported models or training checkpoints while training continues
(it only needs a few GB of GPU memory); it never slows the training loop down.

Two test suites, both on text the model never trained on:

1. In-domain: held-out sentences from valid.jsonl, cloned from held-out recordings of the
   training speakers (one prompt per speaker, from a different segment than the sentences).
     - CER   ASR transcript of the audio vs the sentence (MMS CTC aligner model, char level)
     - SIM   WavLM x-vector cosine to other real clips of the prompt speaker (as kyutai's eval)
     - UTMOS audio quality (1-5), each clip scored at its true length (as kyutai's eval)
     - RTF   generation time / audio duration: < 1 is faster than real time
2. Voice cloning: voices that are NOT in the training data (default: 10 English/French
   voices from the public kyutai/tts-voices repo, both genders).
     - SIM to prompt           how much the output sounds like the prompt voice (higher = better)
     - SIM to training speaker similarity to the nearest training speaker (lower = better)
     - collapsed               share of outputs closer to a training speaker than to the prompt

The real recordings are scored the same way as a ceiling (--no-ceiling to skip).

Run from the pocket-tts checkout:
    uv run python ../recipes/eval_checkpoints.py \\
        --model "teacher 24L 5k=../pocket-tts-moore-step5000/moore_local.yaml" \\
        --model "student 6L 30k=../pocket-tts-moore-6l-step30000/moore_local.yaml" \\
        --model "student 6L 200k=../recipes/moore/moore_6l_infer.yaml@../runs/moore_fr6l_distill/checkpoint_00200000.pt" \\
        --data ../data/moore --aligner ../data/models/mms-1b-mos --out ../runs/eval_moore.json

A model is NAME=CONFIG (an exported *_local.yaml) or NAME=CONFIG@CHECKPOINT (an inference
config plus a training checkpoint .pt).
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from pocket_tts import TTSModel
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor, WavLMForXVector

DEFAULT_CLONE_VOICES = [
    "vctk/p225_023_enhanced.wav", "vctk/p228_023_enhanced.wav", "vctk/p229_023_enhanced.wav",
    "alba-mackenna/casual.wav", "vctk/p226_023_enhanced.wav", "vctk/p227_023_enhanced.wav",
    "cml-tts/fr/10087_11650_000028-0002_enhanced.wav", "cml-tts/fr/10177_10625_000134-0003_enhanced.wav",
    "cml-tts/fr/10179_11051_000005-0001_enhanced.wav", "cml-tts/fr/12977_10625_000037-0001_enhanced.wav",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, help="NAME=CONFIG or NAME=CONFIG@CHECKPOINT (repeatable)")
    p.add_argument("--data", required=True, help="folder with train.jsonl and valid.jsonl (recipes/prepare_hf_dataset.py)")
    p.add_argument("--aligner", required=True, help="character-level Wav2Vec2ForCTC for the language (export_mms_aligner.py)")
    p.add_argument("--n-sentences", type=int, default=24)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 100])
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=110)
    p.add_argument("--max-speakers", type=int, default=3, help="in-domain prompts: the N training speakers with most data")
    p.add_argument("--clone-voices", nargs="*", default=DEFAULT_CLONE_VOICES,
                   help="files in kyutai/tts-voices or local paths; empty to skip the cloning suite")
    p.add_argument("--clone-sentences", type=int, default=2)
    p.add_argument("--no-ceiling", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write all results as JSON")
    p.add_argument("--seed", type=int, default=7, help="sentence sampling seed")
    return p.parse_args()


def norm_text(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", t.lower())).strip()


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


class Scorers:
    def __init__(self, aligner, device):
        self.device = device
        self.asr = Wav2Vec2ForCTC.from_pretrained(aligner).to(device).eval()
        self.proc = Wav2Vec2Processor.from_pretrained(aligner)
        self.spk_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.spk = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        try:
            from utmos_pytorch import UTMOSScoreTorch
            self.utmos = UTMOSScoreTorch(device=str(device))
        except ImportError:
            self.utmos = None
            print("utmos_pytorch not installed: UTMOS skipped")

    @staticmethod
    def to16k(wav, sr):
        return torchaudio.functional.resample(torch.as_tensor(wav, dtype=torch.float32), sr, 16000).numpy()

    def load16k(self, path):
        w, sr = sf.read(path, dtype="float32", always_2d=True)
        return self.to16k(w.mean(1), sr)

    @torch.no_grad()
    def transcribe(self, w16):
        x = self.proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        return self.proc.batch_decode(self.asr(x).logits.argmax(-1))[0]

    @torch.no_grad()
    def xvec(self, w16):
        inp = self.spk_fe([w16], sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
        return torch.nn.functional.normalize(self.spk(**inp).embeddings[0], dim=-1)

    def centroid(self, paths):
        return torch.nn.functional.normalize(torch.stack([self.xvec(self.load16k(p)) for p in paths]).mean(0), dim=-1)

    @torch.no_grad()
    def mos(self, w16):
        if self.utmos is None:
            return float("nan")
        return float(self.utmos.score(torch.from_numpy(w16)[None, None].to(self.device)))


def build_suites(args, sc):
    data = Path(args.data)
    train = [json.loads(l) for l in open(data / "train.jsonl")]
    valid = [json.loads(l) for l in open(data / "valid.jsonl")]
    pairs = [(s, r) for r in valid for s in re.split(r"(?<=[.!?])\s+", r["transcript"])
             if args.min_chars < len(s) < args.max_chars]
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n_sentences]
    sources = {r["path"] for _, r in pairs}

    hours = {}
    for r in train:
        hours[r["speaker"]] = hours.get(r["speaker"], 0) + r["duration"]
    top = sorted(hours, key=hours.get, reverse=True)[: args.max_speakers]

    in_domain = {}
    for spk in top:
        cands = [r for r in valid if r["speaker"] == spk and r["path"] not in sources and r["duration"] >= 8]
        if not cands:  # fall back to training audio for speakers absent from valid
            cands = [r for r in train if r["speaker"] == spk and r["duration"] >= 8]
        prompt, refs = cands[0]["path"], [r["path"] for r in cands[1:6]]
        if len(refs) < 3:
            refs += [r["path"] for r in random.Random(0).sample([r for r in train if r["speaker"] == spk], 5)]
        in_domain[spk] = {"prompt": prompt, "ref": sc.centroid(refs)}
    train_centroids = {spk: sc.centroid([r["path"] for r in random.Random(0).sample([r for r in train if r["speaker"] == spk], 6)])
                       for spk in top}

    clone = {}
    for v in args.clone_voices:
        path = v if Path(v).exists() else hf_hub_download("kyutai/tts-voices", v)
        clone[v] = {"prompt": path, "emb": sc.xvec(sc.load16k(path))}
    return pairs, in_domain, train_centroids, clone


def ceiling(pairs, in_domain, sc):
    err = chars = 0
    mos = []
    for path in sorted({r["path"] for _, r in pairs}):
        r = next(r for _, r in pairs if r["path"] == path)
        w = sc.load16k(path)
        ref = norm_text(r["transcript"])
        err += edit_distance(ref, norm_text(sc.transcribe(w)))
        chars += len(ref)
        mos.append(sc.mos(w))
    sims = {spk: float(sc.xvec(sc.load16k(d["prompt"])) @ d["ref"]) for spk, d in in_domain.items()}
    return {"cer": err / chars, "sim": float(np.mean(list(sims.values()))), "utmos": float(np.nanmean(mos))}


def evaluate(name, spec, pairs, in_domain, train_centroids, clone, args, sc):
    cfg, _, ckpt = spec.partition("@")
    model = TTSModel.load_model(config=cfg, checkpoint=ckpt or None)
    model.to(args.device)
    sr = model.config.mimi.sample_rate
    cuda = str(args.device).startswith("cuda")
    sents = [s for s, _ in pairs]

    def gen(state, text, seed):
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        a = model.generate_audio(state, text).detach().float().cpu().numpy()
        if cuda:
            torch.cuda.synchronize()
        return sc.to16k(a, sr), time.perf_counter() - t, len(a) / sr

    err = chars = 0
    sims, moss, wall, audio = [], [], 0.0, 0.0
    for spk, d in in_domain.items():
        state = model.get_state_for_audio_prompt(d["prompt"])
        gen(state, sents[0], 0)  # warm-up
        for seed in args.seeds:
            for k, s in enumerate(sents):
                w, dt, dur = gen(state, s, seed + k)
                wall += dt
                audio += dur
                ref = norm_text(s)
                err += edit_distance(ref, norm_text(sc.transcribe(w)))
                chars += len(ref)
                sims.append(float(sc.xvec(w) @ d["ref"]))
                moss.append(sc.mos(w))
    res = {"cer": err / chars, "sim": float(np.mean(sims)), "utmos": float(np.nanmean(moss)), "rtf": wall / audio}

    if clone:
        sp, st, col = [], [], []
        for v, d in clone.items():
            state = model.get_state_for_audio_prompt(d["prompt"])
            for k, s in enumerate(sents[: args.clone_sentences]):
                w, _, _ = gen(state, s, k)
                e = sc.xvec(w)
                to_prompt = float(e @ d["emb"])
                to_train = max(float(e @ c) for c in train_centroids.values())
                sp.append(to_prompt)
                st.append(to_train)
                col.append(to_train > to_prompt)
        res.update({"clone_sim_prompt": float(np.mean(sp)), "clone_sim_train": float(np.mean(st)),
                    "clone_collapsed": float(np.mean(col))})
    del model
    if cuda:
        torch.cuda.empty_cache()
    return res


def main():
    args = parse_args()
    sc = Scorers(args.aligner, args.device)
    pairs, in_domain, train_centroids, clone = build_suites(args, sc)
    print(f"{len(pairs)} held-out sentences x {len(args.seeds)} seeds x {len(in_domain)} training-speaker prompts; "
          f"{len(clone)} external voices for the cloning suite", flush=True)

    results = {}
    if not args.no_ceiling:
        results["real recordings"] = ceiling(pairs, in_domain, sc)
    for spec in args.model:
        name, _, rest = spec.partition("=")
        results[name] = evaluate(name, rest, pairs, in_domain, train_centroids, clone, args, sc)
        print(name, {k: round(v, 3) for k, v in results[name].items()}, flush=True)

    head = f"| {'model':30} | CER | SIM | UTMOS | RTF | clone SIM→prompt | clone SIM→train | collapsed |"
    print("\n" + head + "\n|" + "---|" * 8)
    for name, r in results.items():
        f = lambda k, fmt: format(r[k], fmt) if k in r else "–"
        print(f"| {name:30} | {f('cer', '.1%')} | {f('sim', '.3f')} | {f('utmos', '.2f')} | {f('rtf', '.2f')} | "
              f"{f('clone_sim_prompt', '.3f')} | {f('clone_sim_train', '.3f')} | {f('clone_collapsed', '.0%')} |")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
