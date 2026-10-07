"""Turn a Hugging Face speech dataset (audio + transcript [+ speaker, chapter]) into
Pocket TTS manifests (train.jsonl / valid.jsonl / unseen.jsonl) plus 24 kHz wav files.

Steps, in order:
  * Reads the split in dataset order (consecutive rows of a chapter/recording are
    assumed to be consecutive utterances, as in audiobook/Bible corpora).
  * Cleans text: Unicode NFC, non-breaking hyphen -> "-", whitespace collapsed,
    ALL-CAPS lines -> sentence case (--keep-caps to disable), a final "." on lines
    without end punctuation. Rows containing digits are dropped (--keep-digits to
    keep them): digits have no spoken form the aligner/tokenizer can learn unless
    you verbalize them yourself beforehand.
  * Merges consecutive clips of the same speaker (and --group-col value) into
    segments of at most --max-sec seconds (--no-merge to disable). Pocket TTS cuts
    every training sample into a voice prompt (>= 1 s) and a target (>= 1 s) at a
    word boundary, so short clips alone waste data. Clips are concatenated as-is:
    check they carry some edge silence (Bible verses do, ~0.3 s).
  * Resamples everything to --sample-rate (24 kHz, Mimi's rate). Audio recorded at
    a lower rate is upsampled but keeps its original bandwidth.
  * Splits: --holdout-speakers go to unseen.jsonl (test cloning of new voices);
    about --valid-hours of whole groups (chapters) of the other speakers go to
    valid.jsonl; the rest to train.jsonl.

Example (Mooré):
    uv run python ../recipes/prepare_hf_dataset.py --dataset Minervus00/moore_audio_bible \\
        --text-col text --speaker-col speaker_id --group-col chapter \\
        --holdout-speakers SPEAKER_01 --out ../data/moore
"""

import argparse
import io
import json
import random
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm import tqdm

END_PUNCT = ".!?:;,»›\"”"


def clean_text(text: str, fix_caps: bool) -> str:
    text = unicodedata.normalize("NFC", text).replace("‑", "-")
    text = re.sub(r"\s+", " ", text).strip()
    if fix_caps and text.isupper():
        text = text.lower()
        text = text[:1].upper() + text[1:]
    if text and text[-1] not in END_PUNCT:
        text += "."
    return text


def build_segments(rows: list[dict], args) -> tuple[list[dict], dict]:
    stats = defaultdict(int)
    segments, cur = [], None

    def flush():
        nonlocal cur
        if cur is not None:
            segments.append(cur)
        cur = None

    for r in rows:
        if not args.keep_digits and any(c.isdigit() for c in r["text"]):
            stats["dropped_digits"] += 1
            flush()
            continue
        if r["duration"] > args.max_clip_sec or r["duration"] < args.min_clip_sec:
            stats["dropped_duration"] += 1
            flush()
            continue
        text = clean_text(r["text"], fix_caps=not args.keep_caps)
        if not text:
            stats["dropped_empty"] += 1
            flush()
            continue
        same_run = cur is not None and cur["speaker"] == r["speaker"] and cur["group"] == r["group"]
        if not args.no_merge and same_run and cur["duration"] + r["duration"] <= args.max_sec:
            cur["rows"].append(r["index"])
            cur["texts"].append(text)
            cur["duration"] += r["duration"]
        else:
            flush()
            cur = {"speaker": r["speaker"], "group": r["group"], "rows": [r["index"]], "texts": [text], "duration": r["duration"]}
    flush()
    return segments, dict(stats)


_DS = None
_AUDIO_COL = None
_SR = None


def _init_worker(ds, audio_col: str, sample_rate: int) -> None:
    global _DS, _AUDIO_COL, _SR
    _DS, _AUDIO_COL, _SR = ds, audio_col, sample_rate
    torch.set_num_threads(1)


def _load(index: int) -> np.ndarray:
    audio = _DS[index][_AUDIO_COL]
    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != _SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, _SR).numpy()
    return wav


def _write_segment(job: tuple[str, list[int]]) -> float:
    out_path, rows = job
    wav = np.concatenate([_load(i) for i in rows])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.clip(wav, -1.0, 1.0), _SR, subtype="PCM_16")
    return len(wav) / _SR


def safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", str(value)) or "none"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="HF dataset id (or local path) for datasets.load_dataset")
    p.add_argument("--config", default=None, help="dataset config name, if any")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--audio-col", default="audio")
    p.add_argument("--text-col", default="text")
    p.add_argument("--speaker-col", default=None, help="speaker column; without it all rows are one speaker")
    p.add_argument("--group-col", default=None, help="chapter/recording column; merging never crosses groups")
    p.add_argument("--pseudo-group-size", type=int, default=0,
                   help="without --group-col: treat every N consecutive rows as one group, so merging and the "
                        "valid split (which holds out whole groups) still work; 0 = a single group")
    p.add_argument("--duration-col", default="duration", help="seconds; computed from audio if the column is missing")
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--max-sec", type=float, default=15.0, help="merge consecutive clips up to this length")
    p.add_argument("--min-clip-sec", type=float, default=0.3)
    p.add_argument("--max-clip-sec", type=float, default=30.0, help="drop single clips longer than this")
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--keep-digits", action="store_true")
    p.add_argument("--keep-caps", action="store_true")
    p.add_argument("--holdout-speakers", default="", help="comma-separated speaker ids written to unseen.jsonl")
    p.add_argument("--valid-hours", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (for testing)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out).resolve()
    ds = load_dataset(args.dataset, args.config, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    ds = ds.cast_column(args.audio_col, Audio(decode=False))

    texts = ds[args.text_col]
    speakers = ds[args.speaker_col] if args.speaker_col else ["spk"] * len(ds)
    if args.group_col:
        groups = ds[args.group_col]
    elif args.pseudo_group_size > 0:
        groups = [f"g{i // args.pseudo_group_size:04d}" for i in range(len(ds))]
    else:
        groups = [None] * len(ds)
    if args.duration_col in ds.column_names:
        durations = [float(d) for d in ds[args.duration_col]]
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(ds, args.audio_col, args.sample_rate)) as ex:
            durations = [len(w) / args.sample_rate for w in tqdm(ex.map(_load, range(len(ds)), chunksize=64), total=len(ds), desc="durations")]
    rows = [
        {"index": i, "text": t or "", "speaker": str(s), "group": g, "duration": d}
        for i, (t, s, g, d) in enumerate(zip(texts, speakers, groups, durations))
    ]

    segments, stats = build_segments(rows, args)
    print(f"{len(rows)} clips ({sum(durations) / 3600:.1f} h) -> {len(segments)} segments; {stats}")

    holdout = {s for s in args.holdout_speakers.split(",") if s}
    unknown = holdout - {r["speaker"] for r in rows}
    assert not unknown, f"--holdout-speakers not found in the data: {unknown}"
    keys = sorted({(s["speaker"], str(s["group"])) for s in segments if s["speaker"] not in holdout})
    random.Random(args.seed).shuffle(keys)
    key_hours = defaultdict(float)
    for s in segments:
        key_hours[(s["speaker"], str(s["group"]))] += s["duration"] / 3600
    valid_keys, valid_h = set(), 0.0
    for key in keys:
        if valid_h >= args.valid_hours:
            break
        valid_keys.add(key)
        valid_h += key_hours[key]

    jobs, records = [], {"train": [], "valid": [], "unseen": []}
    counters = defaultdict(int)
    for s in segments:
        key = (s["speaker"], str(s["group"]))
        split = "unseen" if s["speaker"] in holdout else "valid" if key in valid_keys else "train"
        counters[key] += 1
        path = out / "wavs" / safe_name(s["speaker"]) / f"{safe_name(s['group'])}_{counters[key]:03d}.wav"
        jobs.append((str(path), s["rows"]))
        records[split].append(
            {"path": str(path), "duration": None, "transcript": " ".join(s["texts"]), "speaker": s["speaker"],
             "group": s["group"], "n_clips": len(s["rows"])}
        )

    durations_written = {}
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(ds, args.audio_col, args.sample_rate)) as ex:
        for (path, _), dur in zip(jobs, tqdm(ex.map(_write_segment, jobs, chunksize=32), total=len(jobs), desc="wav")):
            durations_written[path] = dur

    for split, recs in records.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for rec in recs:
                rec["duration"] = round(durations_written[rec["path"]], 4)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        per_spk = defaultdict(float)
        for r in recs:
            per_spk[r["speaker"]] += r["duration"] / 3600
        hours = sum(per_spk.values())
        print(f"{split}: {len(recs)} segments, {hours:.1f} h, per speaker: { {k: round(v, 1) for k, v in sorted(per_spk.items())} }")


if __name__ == "__main__":
    main()
