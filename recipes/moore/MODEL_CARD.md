---
license: cc-by-4.0
language:
  - mos
base_model: kyutai/pocket-tts
pipeline_tag: text-to-speech
library_name: pocket-tts
tags:
  - text-to-speech
  - voice-cloning
  - moore
  - mossi
  - burkina-faso
  - pocket-tts
---

# Pocket TTS Mooré (mos) — 24-layer, step 5,000

A text-to-speech model for **Mooré** (Mossi, ISO 639-3 `mos`) with zero-shot **voice cloning** from a short audio prompt.
It is a fine-tune of Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) French 24-layer teacher (`french_24l`),
trained with the official new-language recipe (`training/configs/finetune_language.yaml`): new Mooré tokenizer, text embedding re-initialised, full fine-tune.

> **Status: best-sounding model of the run (teacher step 5,000, chosen at the validation-loss minimum; later checkpoints
> overfit).** Good intelligibility on Mooré read speech, but **voice cloning is limited to voices close to the training
> narrators**: see [Evaluation](#evaluation) and [Limitations](#limitations).

## Usage

Requires the `pocket-tts` package (`pip install pocket-tts` or `uvx pocket-tts`). Text is passed **as written in Mooré orthography**; there is no text normalisation (see [Text input](#text-input)).

```bash
uvx pocket-tts generate \
  --config hf://REPLACE_WITH_HF_USER/pocket-tts-moore/moore.yaml \
  --voice reference_voice.wav \
  --text "A Abimelɛk wãa tẽngã, n bao yamsem n fẽrg a tẽn-gãongã." \
  --output-path out.wav
```

### Inference script

`inference.py` (in this repo) synthesizes one sentence or a whole text file, building the voice state once and reusing it:

```bash
pip install pocket-tts scipy

# one sentence -> output.wav
python inference.py --voice reference_voice.wav \
  --text "A Abimelɛk wãa tẽngã, n bao yamsem n fẽrg a tẽn-gãongã." --output output.wav

# one sentence per line of a UTF-8 file -> out/0000.wav, out/0001.wav, ...
python inference.py --voice reference_voice.wav --text-file sentences.txt --output out/ --device cuda
```

It writes 16-bit, 24 kHz mono wav files and NFC-normalizes the text. Options: `--device cpu|cuda`, `--temperature` (default 0.3), `--config` (defaults to `moore.yaml`, i.e. weights from the Hub; use `--config moore_local.yaml` for local files).

### Python

```python
import numpy as np
import scipy.io.wavfile
from pocket_tts import TTSModel

model = TTSModel.load_model(config="hf://REPLACE_WITH_HF_USER/pocket-tts-moore/moore.yaml")
model.to("cpu")  # or "cuda"
state = model.get_state_for_audio_prompt("reference_voice.wav")  # 3-10 s of clean speech
audio = model.generate_audio(state, "La baa ne rẽ fãa, a Zehu ka sak Sẽn-Ka-Saab.")  # 1-D float tensor

pcm16 = (np.clip(audio.cpu().numpy(), -1, 1) * 32767).astype(np.int16)
scipy.io.wavfile.write("output.wav", model.config.mimi.sample_rate, pcm16)  # 24 kHz
```

Reuse `state` for more sentences in the same voice.

- `--voice`: a clean 3–10 s recording of the target speaker. The output mimics its acoustics, so noisy prompts give noisy output.
- Speed (RTF = generation time / audio duration, < 1 = faster than real time): ~0.7 on a shared A100 GPU, ~1.2–2 on CPU (2.2 GHz Xeon). The 6-layer distilled student is ~3× faster (RTF ~0.27 GPU, ~0.55 CPU).
- To use the files locally before uploading, use `moore_local.yaml` (absolute paths to this folder) instead of `moore.yaml`.

### Text input

- Standard Mooré orthography, Unicode **NFC** (ã ẽ ĩ õ ũ ɛ ɩ ʋ are single code points).
- **Write numbers in words.** Utterances containing digits were removed from training, so digits are unseen.
- Punctuation (`. , ! ? : ; « »` and `-`) was kept in training and influences pauses.
- Headings in ALL CAPS were converted to sentence case in training; prefer normal casing.

## Samples

In `samples/` (24 kHz wav):

| File | Content |
|---|---|
| `step00005000_0.wav` | "La baa ne rẽ fãa, a Zehu ka sak Sẽn-Ka-Saab sẽn yaa Israyɛll Wẽnnaamã tõodã ne a sũur fãa ye." |
| `step00005000_1.wav` | "A Abimelɛk wãa tẽngã, n bao yamsem n fẽrg a tẽn-gãongã." |
| `step00005000_2.wav` | "La Amo-rãmbã sẽn yã tɩ Siiri nebã zoeeme wã, bãmb me zoe a Abisayi ne a tãb-biisã taoor n kẽ tẽngã pʋgẽ." |
| `unseen_speaker01_clone.wav` | Sentence 1, voice prompt from the held-out split (SPEAKER_01), generated from this folder with the CLI on CPU |

All three sentences come from chapters held out of training. The `step*` samples were generated during training (voice prompt from a training batch, CFG 2.0).

## Evaluation

**Intelligibility, similarity and quality.** 24 held-out sentences (validation chapters) × 2 sampling seeds, each generated
from two voice prompts taken from held-out chapters of two training narrators (SPEAKER_03 and SPEAKER_04), on GPU.
- **CER:** transcripts by [MMS-1b-all](https://huggingface.co/facebook/mms-1b-all) (`mos` adapter).
- **Speaker similarity (SIM):** WavLM x-vector cosine to other real clips of the prompt speaker, as in Kyutai's evaluation.
- **UTMOS:** automatic quality estimate. It was trained on English, so compare models with each other rather than with the real recordings.
- **RTF:** generation time / audio duration on a shared A100 (< 1 = faster than real time).

The real recordings are scored the same way, as the ceiling.

| Model | CER | SIM (SPEAKER_04 / SPEAKER_03) | UTMOS | RTF (GPU) |
|---|---|---|---|---|
| real recordings | 8.4% | 0.996 / 0.988 | 2.65 | – |
| teacher 24L, step 5,000 | **13.1%** | 0.946 / 0.982 | 3.53 | 0.71 |
| student 6L, step 30,000 | 14.9% | 0.940 / 0.982 | 3.48 | **0.27** |
| student 6L, step 40,000 | 15.9% | 0.944 / 0.979 | 3.51 | 0.27 |
| student 6L, step 200,000 (end of distillation) | 18.9% | 0.941 / 0.980 | 3.50 | 0.27 |
| student 6L, step 200,000, EMA weights | 19.1% | 0.939 / 0.979 | 3.49 | 0.27 |

Longer distillation fitted the teacher better (validation loss 0.0175 at 30k → 0.0154 at 200k), but made intelligibility worse,
and speaker similarity and quality stayed flat.

**Voice cloning of new voices.** 10 voices that are not in the training data (4 female and 2 male English voices from
VCTK/Alba, 4 French CML-TTS readers, from `kyutai/tts-voices`), 2 sentences each. Kyutai's original French models, reading
French text, are the reference for what cloning looks like when it works.

| Model | SIM to the prompt voice (higher = better) | SIM to the nearest training narrator | outputs closer to a training narrator |
|---|---|---|---|
| Kyutai french 24L, original | 0.84 | 0.71 | 35% |
| Kyutai french 6L, original | 0.84 | 0.71 | 30% |
| teacher 24L, step 5,000 | 0.71 | 0.88 | **85%** |
| student 6L, step 30,000 | 0.68 | 0.93 | **95%** |

**This model does not really clone new voices.** Most outputs sound like one of the training narrators, whatever the prompt.
See [Limitations](#limitations).

## Training

| | |
|---|---|
| Base model | `kyutai/pocket-tts`, `languages/french_24l` (24-layer teacher, with voice cloning) |
| Recipe | `finetune_language.yaml`: `reset_text_embedding: true`, LSD flow objective, `flow_batch_multiplier: 4` |
| Tokenizer | SentencePiece unigram, 4,000 pieces, character coverage 1.0, trained on the Mooré transcripts (`tokenizer.model`; `tokenizer.json` is the exact conversion the model loads) |
| Optimiser | AdamW, lr 2e-4 constant after 1,000 warmup steps, weight decay 0.1, grad clip 1.0 |
| Batch | 16 × 4 gradient accumulation = 64 rows per step |
| Steps | **5,000** (≈ 15 passes over the training set) |
| Weights | raw weights at step 5,000, the ones the `samples/` were generated with (not the EMA shadow) |
| Hardware | 1 × A100-40GB, shared with another job; ~1.4 steps/s |
| Validation | `flow_diag` on 1.0 h of held-out chapters: 10.45 (step 2,500) → **10.38 (step 5,000, minimum)** → 11.01 (7,500) → 12.24 (10,000) |

Full resolved configuration: `training_args.yaml`.

### Data

Mooré Bible read speech (`Minervus00/moore_audio_bible`, 57,410 utterances, 74.3 h, 24 kHz, 5 speakers).

- Removed 932 utterances containing digits and 70 longer than 30 s.
- Consecutive verses of the same speaker and chapter merged into segments of ≤ 15 s (median 12.7 s), so that each training sample can be split into a voice prompt and a target at a word boundary.
- Word alignments by CTC forced alignment with MMS-1b-all (`mos` adapter).
- Splits: **train 71.1 h** (SPEAKER_00 35.3 h, SPEAKER_03 23.1 h, SPEAKER_04 11.7 h, SPEAKER_02 1.0 h); **validation 1.0 h** (held-out chapters); **held out 0.3 h** (all of SPEAKER_01).

## Limitations

- **Small dataset, early overfitting.** With 71 h of training audio, validation loss was lowest at step 5,000 and rose afterwards; this checkpoint is the validation minimum.
- **Voice cloning is limited to the training narrators.** 97% of the training audio comes from 3 male narrators. Fine-tuning on so few voices made the model largely ignore the voice prompt: with voices outside the training data (including female voices), 85% (teacher) to 95% (student) of the outputs sound closer to a training narrator than to the prompt. Expect a Bible-reading style too.
- **24 kHz, 24-layer teacher.** Not distilled to the 6-layer student, so slower than the official CPU-oriented Pocket TTS models.
- No numbers, abbreviations or code-switching (e.g. French words) in training.

## Files

| File | Purpose |
|---|---|
| `model.safetensors` | flow LM + Mimi codec weights, loaded by `pocket-tts` via the config |
| `tokenizer.json` | tokenizer loaded by the model |
| `tokenizer.model` | SentencePiece source of the tokenizer |
| `inference.py` | command-line synthesis script (single sentence or text file) |
| `moore.yaml` | model config for use from the Hub (`hf://…` paths) |
| `moore_local.yaml` | same config with local absolute paths (for testing; do not upload) |
| `training_args.yaml` | training configuration |
| `checkpoint_00005000.pt` | raw training checkpoint (model + EMA), for resuming or re-exporting; optional to upload |
| `samples/` | audio samples |

## License, attribution and use restrictions

- Derived from **Kyutai's Pocket TTS** weights, released under **CC-BY-4.0**. Attribution: Kyutai, [Pocket TTS](https://github.com/kyutai-labs/pocket-tts).
- Kyutai's use policy applies to this derivative. Prohibited uses include **voice impersonation or cloning without explicit and lawful consent**, misinformation or deception (including presenting generated audio as genuine recordings of real people), and generating unlawful, harmful, harassing, discriminatory or privacy-invasive content. Only clone voices you have permission to use.
- **Training data rights: to confirm before public release.** The source dataset declares no license and is access-gated; Bible audio recordings are often under publisher copyright.

## Before uploading

1. Replace `REPLACE_WITH_HF_USER/pocket-tts-moore` in `moore.yaml` and in this README with the real repo id.
2. Confirm the training-data rights (above), and consider gating the repo like `kyutai/pocket-tts`.
3. Upload, leaving out the local-only files:
   ```bash
   huggingface-cli upload <user>/pocket-tts-moore . . --exclude "moore_local.yaml" "*.pt"   # drop "*.pt" from --exclude to include the training checkpoint
   ```
4. Test from the Hub: `uvx pocket-tts generate --config hf://<user>/pocket-tts-moore/moore.yaml --voice voice.wav --text "…"`.
