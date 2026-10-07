# Fine-tuning Pocket TTS for new languages (worked examples: Mooré and Dyula)

Recipes and scripts to teach [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) a language it does not
speak, then distil it into the small CPU-friendly model, and optionally adapt it to one speaker's voice.
Everything was run end to end for two languages of Burkina Faso and Côte d'Ivoire, with ~70 h of read speech each:

- **Mooré** (Mossi, `mos`): 74 h, 24 kHz, recipe in [`recipes/moore/`](recipes/moore)
- **Dyula** (Jula, `dyu`): 70 h, 16 kHz, recipe in [`recipes/dyula/`](recipes/dyula), plus a single-speaker female voice

The same steps apply to any language with transcribed speech: see [Adapting to another language](#adapting-to-another-language).

```
HF dataset ──prepare──► 24 kHz segments + manifests ──tokenizer──► SentencePiece (4k)
                                      │
                                      └──align (MMS CTC)──► word timestamps
                                                                 │
        kyutai french_24l ──fine-tune (new text embedding)──► 24-layer teacher ──distil──► 6-layer student
                                                                 │                              │
                                                                 └──── evaluate · export ───────┴──► HF model folder
```

## Contents

| Path | What it is |
|---|---|
| `recipes/prepare_hf_dataset.py` | HF dataset → cleaned text, merged segments, 24 kHz wavs, `train/valid/unseen.jsonl` |
| `recipes/export_mms_aligner.py` | Exports [MMS-1b-all](https://huggingface.co/facebook/mms-1b-all) with one language adapter as a plain CTC model, for forced alignment and CER evaluation |
| `recipes/eval_checkpoints.py` | Compares checkpoints on held-out sentences (CER, speaker similarity, UTMOS, RTF) plus a voice-cloning test on external voices; runs on its own, so it never slows training |
| `recipes/export_release.py` | Packages a training checkpoint as a Hugging Face–ready Pocket TTS model folder |
| `recipes/inference.py` | Standalone synthesis script copied into each exported model folder |
| `recipes/listen_samples.ipynb` | Plays the training samples across steps next to the real recordings, and plots the validation curves |
| `recipes/moore/` | Mooré recipe: `run_moore.sh` (all stages), teacher/student configs, inference configs, model card template |
| `recipes/dyula/` | Dyula recipe: `run_dyula.sh`, teacher/student configs, inference configs, and the female-voice configs |
| `push_to_hub.py` | Uploads an exported model folder (or a .zip/.tar.gz of one) to the Hugging Face Hub |
| `inference_kaggle.ipynb` | Kaggle-style notebook: download a model from the Hub, CPU vs GPU inference, voice cloning from a Google Drive sample |
| `.env.example` | Template for `.env` (`HF_TOKEN`, `HF_USERNAME`), read by `push_to_hub.py` and the `run_*.sh` scripts |

Not in git (see `.gitignore`): `.env`, the `pocket-tts/` checkout, `data/`, `runs/`, exported model folders, weights, audio and archives.

## Why Pocket TTS

Before settling on Pocket TTS we compared CosyVoice 1/2/3, VoxCPM 0.5B/1.5/2 and Pocket TTS for languages none of them support:

- **CosyVoice** relies on a frozen discrete speech tokenizer trained on ~9 languages. Whether it represents an unseen tonal language well is the main risk, and new-language fine-tuning is undocumented.
- **VoxCPM2** (2B) has no discrete-token bottleneck and good docs, but a full fine-tune needs ~40 GB of VRAM, and its docs recommend 500+ h for a new language.
- **Pocket TTS** ships training code with an official **new-language recipe** (`finetune_language.yaml`: new tokenizer, re-initialised text embedding). It fits a single 40 GB GPU and trains fast, and its distilled 6-layer student runs faster than real time on CPU. Kyutai recommends ≥ 100 h; ~70 h per language worked here, with early overfitting.

## Requirements

- Linux and one NVIDIA GPU. Measured on an A100-40GB:

  | Stage | GPU memory | Speed |
  |---|---|---|
  | Teacher fine-tune (batch 16 × 4, `torch.compile`) | ~27 GB | ~1.8 steps/s (GPU alone), ~1.4 when shared |
  | Distillation (batch 32 × 2) | ~15 GB, ~23 GB at validation | ~3–3.9 steps/s |

  With less memory, lower `batch_size` and raise `grad_accum_steps`: keep **batch_size × grad_accum_steps = 64** (below 64 rows, Kyutai reports the quality transition arrives late or never).
- Disk: ~170 MB per hour of 24 kHz 16-bit audio (~12 GB for 70 h), ~0.5 GB of precomputed latents, 3.6 GB per MMS aligner, 6 GB for the Python environment. A teacher checkpoint is 2.5 GB, a student checkpoint 0.7 GB; each run keeps `num_ckpt_keep` of them plus one optimizer state.
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Gated weights and datasets:** accept the terms of [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) (the voice-cloning weights) and of your datasets on Hugging Face. Then either put a token in `.env` (`cp .env.example .env`; the `run_*.sh` scripts load it) or run `huggingface-cli login`.

## Setup

```bash
git clone <this repo> ft-pocket-tts && cd ft-pocket-tts
cp .env.example .env                        # fill in HF_TOKEN (and HF_USERNAME for uploads)
git clone https://github.com/kyutai-labs/pocket-tts.git
cd pocket-tts
git checkout 3dbee45                        # commit these recipes were run and tested with
uv sync                                     # training + eval deps (dev group); torch with CUDA from PyPI
uv pip install "datasets>=3,<5" soundfile   # used by recipes/prepare_hf_dataset.py
uv run pytest training/tests -q             # 69 tests should pass
cd ..
bash recipes/moore/run_moore.sh check       # verifies access to the gated kyutai/pocket-tts weights
```

If `torch.cuda.is_available()` is `False`, your driver is older than the CUDA build uv installed: see "Using older cuda versions" in `pocket-tts/training/README.md`.

All configs use paths **relative to the `pocket-tts/` checkout**; the `run_*.sh` scripts `cd` there themselves. Every stage is idempotent (finished outputs are skipped) and runs from the repo root.

## Reproducing the Mooré model

### 1. Prepare the data (~1 min for 74 h with 8 workers, plus download)

```bash
bash recipes/moore/run_moore.sh prep
```

Runs `recipes/prepare_hf_dataset.py` on `Minervus00/moore_audio_bible` (57,410 clips, 74.3 h, 24 kHz), then trains the tokenizer:

- **Text cleaning:** Unicode NFC, non-breaking hyphen → `-`, ALL-CAPS headings → sentence case, final `.` added when missing. **Rows containing digits are dropped** (932 here, mostly chapter headings): digits have no spoken form the model can learn unless you verbalise them yourself.
- **Merging:** consecutive clips of the same speaker and chapter are concatenated into segments of ≤ 15 s (median 3.2 s → 12.7 s). Pocket TTS cuts every sample into a voice prompt and a target at a word boundary, with ≥ 1 s on each side, so short clips are mostly unusable alone. Bible verses already carry ~0.3 s of edge silence, so no padding is inserted.
- **Splits:** `unseen.jsonl` = every clip of `SPEAKER_01` (0.3 h, kept out of training); `valid.jsonl` = ~1 h of whole held-out chapters; `train.jsonl` = the rest (71.1 h).
- **Tokenizer:** SentencePiece unigram, 4,000 pieces (must equal the model's `n_bins`), character coverage 1.0, then converted to the `tokenizer.json` the model loads. The conversion is checked against SentencePiece on the validation texts.

### 2. Word alignments (minutes)

```bash
bash recipes/moore/run_moore.sh align
```

Pocket TTS needs word timestamps to cut prompt and target at word boundaries. `export_mms_aligner.py` saves MMS-1b-all with the `mos` adapter merged, and the upstream `training/scripts/align_data.py` runs CTC forced alignment with it. The aligner lowercases and drops characters outside its alphabet per word, so punctuation needs no special handling. Result: 21,201 / 21,206 train segments aligned.

### 3. Fine-tune the 24-layer teacher

```bash
bash recipes/moore/run_moore.sh train fr24l     # background (nohup); log: runs/moore_fr24l.out
tail -f runs/moore_fr24l.out
```

`finetune_moore_fr24l.yaml` starts from Kyutai's **French** 24-layer model (`finetune_moore_en24l.yaml` from the English one) with `reset_text_embedding: true`, lr 2e-4 constant, 1,000 warm-up steps and an effective batch of 64. It auto-resumes from the latest checkpoint in its run folder if restarted (delete the folder to start over). On the first launch it precomputes Mimi latents for the train set (~3 min).

**Monitor it:**
- Copy `recipes/listen_samples.ipynb` into `runs/moore_fr24l/samples/` and run it. Every 2,500 steps, training renders the 3 `sample_sentences` (taken from held-out chapters); the notebook plays them next to the real recordings and plots the losses.
- **Validation `flow_diag`** (in the log at every validation) is the number to watch: it should fall, then flatten. `loss` includes a learned uncertainty term, so it goes negative and keeps falling even after quality has stopped improving.
- **When validation `flow_diag` rises while training keeps falling, the model is overfitting.** There is no early stopping, and only the last `num_ckpt_keep` checkpoints are kept, so package the best one (step 5) before it rotates out.
- Stop the run by PID: `pgrep -af "python -m training.train"` then `kill <pid>`. It writes a final checkpoint before exiting.

### 4. Evaluate checkpoints

```bash
cd pocket-tts
uv run python ../recipes/eval_checkpoints.py \
  --model "teacher 5k=../recipes/moore/moore_24l_infer.yaml@../runs/moore_fr24l/checkpoint_00005000.pt" \
  --model "teacher 7.5k=../recipes/moore/moore_24l_infer.yaml@../runs/moore_fr24l/checkpoint_00007500.pt" \
  --data ../data/moore --aligner ../data/models/mms-1b-mos --out ../runs/eval_moore.json
```

A model is `NAME=CONFIG@CHECKPOINT` (inference config + training checkpoint) or `NAME=CONFIG` (an exported `*_local.yaml`). It needs a few GB of GPU memory and prints a table:

- **In-domain:** 24 held-out sentences × 2 seeds, cloned from held-out recordings of the training speakers. Reports CER (MMS transcript vs sentence), speaker similarity (WavLM x-vector, as in Kyutai's eval), UTMOS and RTF (generation time / audio duration, < 1 = faster than real time).
- **Voice cloning:** 10 external voices from `kyutai/tts-voices`, both genders. Reports similarity to the prompt vs to the nearest training speaker.
- **Ceiling:** the real recordings, scored the same way.

### 5. Package a checkpoint

```bash
bash recipes/moore/run_moore.sh export moore_fr24l 5000 pocket-tts-moore-step5000 <hf_user>/pocket-tts-moore
```

The export runs on CPU, so it can run next to a training job. It writes:
- `model.safetensors`: **raw** weights, the ones the samples were generated with. `--use-ema` exports the EMA instead; early in a run the EMA still contains a share of the initial weights.
- `moore.yaml` (weights from `hf://<repo>`) and `moore_local.yaml` (local paths).
- The tokenizer, `training_args.yaml`, the training checkpoint, and `inference.py`.

Add a model card (`recipes/moore/MODEL_CARD.md` is a complete example) and a `samples/` folder, then test the folder:

```bash
cd pocket-tts
uv run python ../pocket-tts-moore-step5000/inference.py --config ../pocket-tts-moore-step5000/moore_local.yaml \
  --voice ../data/moore/wavs/SPEAKER_01/jol_1_001.wav --text "A Abimelɛk wãa tẽngã, n bao yamsem n fẽrg a tẽn-gãongã." \
  --output test.wav
```

### 6. Distil into the 6-layer student

```bash
# set distill_teacher_weights in recipes/moore/distill_moore_fr6l.yaml to the packaged teacher checkpoint, then
bash recipes/moore/run_moore.sh distill          # refuses to start while a teacher run is training
tail -f runs/moore_fr6l_distill.out
bash recipes/moore/run_moore.sh export moore_fr6l_distill <step> pocket-tts-moore-6l-step<step> <hf_user>/pocket-tts-moore-6l
```

Why this step matters:
- **Speed.** The 24-layer teacher is ~3× slower than the 6-layer student (CPU RTF ~1.2–2 vs ~0.55).
- **CFG.** The teacher's training samples use classifier-free guidance (CFG 2.0), but `pocket-tts generate` runs without it. The distilled student has CFG 2.0 baked in (`distill_cfg_coef: 2.0`).

How it works:
- The student copies every non-backbone tensor from the teacher: the text embedding, the voice-prompt projection, and the flow and end-of-speech heads, which stay frozen. Its 6 layers are seeded from teacher layers 0–2 and 21–23.
- Distillation is teacher-forced: the student learns to match the teacher's backbone output on real recordings.
- Watch validation `distill_mse` (lower is better), but **choose the student with `eval_checkpoints.py`**: the lowest validation loss is not the most intelligible student (see [Results](#results)). The cosine schedule is tied to `max_steps`.

### 7. Upload

```bash
python push_to_hub.py pocket-tts-moore-6l-step30000 --dry-run    # shows the repo, files and sizes
python push_to_hub.py pocket-tts-moore-6l-step30000              # -> <HF_USERNAME>/pocket-tts-moore-6l, private
python push_to_hub.py pocket-tts-moore-step5000 --public --gated auto   # -> <HF_USERNAME>/pocket-tts-moore
```

`push_to_hub.py` needs only `huggingface_hub`. It accepts a folder or a `.zip` / `.tar.gz` archive of one and never modifies your folder. It stages a copy in which:
- the `REPLACE_WITH_HF_USER/...` placeholders are replaced by your repo id;
- absolute local paths (e.g. in `training_args.yaml`) are made relative;
- `*_local.yaml` and the `*.pt` checkpoints are left out (`--include-checkpoint` to keep them).

It then checks the configs point at the target repo, and creates the repo **private by default**. The repo name comes from the folder's `hf://` config paths; use `--repo-name` to publish two checkpoints of the same model side by side.

### 8. Try it (Kaggle)

Upload `inference_kaggle.ipynb` to Kaggle (GPU accelerator and Internet on; add `HF_TOKEN` under *Add-ons → Secrets* if the repo is private or gated). Set:
- `HF_USERNAME` and `REPOS` (the repo names you pushed)
- `MODEL = "6l"` or `"24l"`
- `VOICE_DRIVE_URL`: a Google Drive share link set to *Anyone with the link*, in any common audio format

It downloads the model and the reference voice (with `gdown`), compares CPU and GPU speed (RTF), runs voice-cloning experiments (all sentences, prompt length 3/5/10 s, temperature, seeds, your own text) and int8 CPU inference, and zips every output. It uses the PyPI `pocket-tts` package (tested with 3.3.0); `LOCAL_MODEL_DIR` and `LOCAL_VOICE` run it on local files instead.

## Reproducing the Dyula model

`recipes/dyula/run_dyula.sh` has the same stages as the Mooré script. Differences in its settings block and configs:

- **Data:** `Minervus00/dyula-speech-bible` (gated; `HF_TOKEN` from `.env`), text column `transcription`, 52,163 clips, 70.2 h at **16 kHz**. Prep resamples to 24 kHz: upsampling does not restore content above 8 kHz.
- **Aligner:** MMS `dyu` adapter (19,950 / 19,955 train segments aligned).
- **Held-out speaker:** `SPEAKER_03` (0.5 h). Splits: train 67.2 h, valid 1.1 h.
- **Checkpoints:** `num_ckpt_keep: 10` in both configs, so the best checkpoint is still on disk when it is chosen.

```bash
bash recipes/dyula/run_dyula.sh prep
bash recipes/dyula/run_dyula.sh align
bash recipes/dyula/run_dyula.sh train fr24l        # best validation around step 2.5-5k: package it, then stop the run
bash recipes/dyula/run_dyula.sh export dyula_fr24l 5000 pocket-tts-dyula-step5000
# set distill_teacher_weights in recipes/dyula/distill_dyula_fr6l.yaml, then
bash recipes/dyula/run_dyula.sh distill
bash recipes/dyula/run_dyula.sh export dyula_fr6l_distill 55000 pocket-tts-dyula-6l-step55000
```

## Adapting a model to one speaker's voice

With a few hours of a single speaker (here ~6 h of a female Dyula speaker, `Minervus00/fem_dyu_16khz`), the teacher can be fine-tuned to speak in that voice, then distilled into a fixed-voice student.

1. **Prepare the speaker's data.** Use the same tokenizer as the base teacher; check that every character of the new transcripts appears in the base training text. This dataset has no chapter column and its rows are shuffled, so `--pseudo-group-size` groups every N consecutive rows for merging and for the held-out split:
   ```bash
   cd pocket-tts
   uv run python ../recipes/prepare_hf_dataset.py --dataset Minervus00/fem_dyu_16khz --text-col text \
     --pseudo-group-size 40 --valid-hours 0.5 --out ../data/dyula_fem
   for s in valid train; do
     uv run python -m training.scripts.align_data ../data/dyula_fem/$s.jsonl ../data/dyula_fem/${s}_aligned.jsonl \
       --model ../data/models/mms-1b-dyu --batch-size 16
   done
   cd ..
   ```
2. **Fine-tune the teacher on the voice** (`finetune_dyula_fem.yaml`, following Kyutai's `finetune.yaml`): starts from the packaged Dyula teacher (`weights_path` override), same tokenizer and text embedding (`reset_text_embedding: false`), lr 2e-5, 200 warm-up steps, validation every 250 steps. With 5.5 h of data (~26 steps per pass), validation is lowest around steps 750–1,000.
   ```bash
   bash recipes/dyula/run_dyula.sh train fem
   bash recipes/dyula/run_dyula.sh export dyula_fem_fr24l 1000 pocket-tts-dyula-fem-step1000 <hf_user>/pocket-tts-dyula-fem
   ```
3. **Distil a fixed-voice student** on the speaker's recordings (`distill_dyula_fem_fr6l.yaml`): the student is seeded from this voice-tuned teacher, so it inherits her voice. A single-speaker student does not clone other voices.
   ```bash
   bash recipes/dyula/run_dyula.sh distill fem_fr6l
   ```

Small datasets need `num_bucket_batches` lowered: each of the `loader_procs` loader processes gets 1/`loader_procs` of the manifest and only yields batches once it holds `num_bucket_batches × batch_size` samples (the fem configs set it to 4).

## Adapting to another language

1. **Check the data.** You need transcribed speech: Kyutai recommends ≥ 100 h (1,000+ for a strong model), and ~70 h worked here. Many speakers help cloning: with 3 main voices, cloning of very different voices is the weak point. Check that clips start and end with some silence (they are concatenated as-is), and prefer hand-made transcripts. A quick check: transcribe a few real clips with the MMS adapter and compute the CER (5–8% here).
2. **Copy a recipe.** `cp -r recipes/moore recipes/<lang>`, rename the files (`moore` → `<lang>`), and edit the block at the top of `run_<lang>.sh`:
   - `LANG_ID`, `HF_DATASET`, `TEXT_COL`, `SPEAKER_COL` and `GROUP_COL` (chapter/recording; leave empty if none, and use `--pseudo-group-size` in `prepare_hf_dataset.py`)
   - `HOLDOUT_SPEAKERS`: one small speaker to test unseen voices
   - `MMS_LANG`
3. **Aligner.** Check that MMS-1b-all has an adapter for your language (its [`vocab.json`](https://huggingface.co/facebook/mms-1b-all/blob/main/vocab.json) is keyed by ISO 639-3 code: `mos` and `dyu` exist) and that the adapter's alphabet covers your orthography. Otherwise use any character-level `Wav2Vec2ForCTC` fine-tuned for the language.
4. **Configs.** In the copied YAMLs, replace the `data/` and `runs/` paths and the `sample_sentences` (take them from `valid.jsonl` so real recordings exist). Choose the starting model: `french_24l` (used here), `english_2026-09_24l`, or another released 24-layer language close to yours. The text embedding is re-initialised in every case; the backbone's acoustic and prosodic knowledge is what transfers. Set `num_ckpt_keep` so the best checkpoint survives until you choose it.
5. **Sample rate.** Everything is resampled to 24 kHz (the Mimi codec's rate). Audio recorded below 24 kHz (e.g. 16 kHz) is upsampled but keeps its original bandwidth.
6. **Text normalisation.** Numbers, abbreviations and symbols must be written out the way they are spoken, at training time *and* at inference. The inference configs here disable the French character replacements inherited from Kyutai's config, because they would strip `« » : ;`, which the training text keeps.
7. **Run** the stages with `run_<lang>.sh`. Validate often; with < 100 h, expect the validation minimum within the first ~5–10k teacher steps.

## Results

Evaluated with `recipes/eval_checkpoints.py` (24 held-out sentences × 2 seeds, MMS CER). UTMOS was trained on English, so compare models with each other rather than with the real recordings. RTF on GPU, < 1 = faster than real time.

**Mooré** (prompts from 2 training narrators)

| Model | CER | Speaker similarity | UTMOS | RTF |
|---|---|---|---|---|
| real recordings | 8.4% | 0.99 | 2.65 | – |
| teacher 24L, step 5k | **13.1%** | 0.95–0.98 | 3.53 | 0.71 |
| student 6L, step 30k | 14.9% | 0.94–0.98 | 3.48 | **0.27** |
| student 6L, step 200k | 18.9% | 0.94–0.98 | 3.50 | 0.27 |

**Dyula** (prompts from the 3 training narrators)

| Model | CER | Speaker similarity | UTMOS | RTF |
|---|---|---|---|---|
| real recordings | 4.7% | 0.982 | 2.55 | – |
| teacher 24L, step 5k | **9.8%** | 0.969 | 3.55 | 0.54 |
| student 6L, step 55k | 13.2% | 0.963 | 3.51 | **0.22** |
| student 6L, step 200k | 22.0% | 0.959 | 3.56 | 0.22 |

- **Distillation:** the most intelligible student comes early (30k for Mooré, 55k for Dyula); longer distillation fits the teacher better but makes intelligibility worse.
- **Voice cloning of new voices does not work well:** on 10 external voices (incl. female), 85–100% of the outputs sound closer to a training narrator than to the prompt, vs 30–35% for Kyutai's original French models. Cloning works within the training narrators' voices (similarity 0.94–0.98).
- **Single-speaker voice fine-tune (Dyula, female):** the voice-tuned teacher at step 1,000 speaks in her voice (speaker similarity 0.975 to her recordings, 0.67 to the nearest narrator).
- **CPU speed:** student RTF ~0.55 (0.35 with int8 `--quantize`), teacher ~1.2–2 (2.2 GHz Xeon VM).

## Lessons learned

- **Speaker diversity drives voice cloning.** With ~97% of the audio from 3 narrators, the fine-tuned model mostly ignores the voice prompt: new voices, female ones in particular, come out sounding like a training narrator. Test cloning on external voices (`eval_checkpoints.py` does), not only on held-out clips of the training speakers. To get a specific voice, fine-tune the teacher on that voice before distilling.
- **Choose checkpoints by evaluation, and keep them.** With ~70 h, the teacher's validation minimum comes within the first ~5k steps, and the most intelligible student well before the end of distillation. Only `num_ckpt_keep` checkpoints are kept: package the good ones as you go.
- **Short clips waste data.** Merging consecutive utterances turned a ~3.3 s median into ~12.7 s, which gives every sample room for a 1–5 s voice prompt.
- **Validation memory scales with `batch_size`, not the effective batch.** Validation re-encodes the validation audio with Mimi at `batch_size`; 64 clips of up to 30 s need a ~9.4 GB allocation. 32 × 2 accumulation gives the same effective batch with half the validation memory.
- **The data loader needs ≥ `num_bucket_batches × batch_size` samples per loader process** before it yields anything (each of the `loader_procs` processes gets 1/`loader_procs` of the manifest). Small datasets and smoke tests need a lower `num_bucket_batches`.
- **Inference has no CFG**: judge the teacher with `pocket-tts generate` as well as with its training samples (rendered with CFG 2.0), or distil it.
- **Transformers 5** removed `Wav2Vec2CTCTokenizer.vocab_file`; `export_mms_aligner.py` reads MMS's `vocab.json` from the Hub instead.

## License and responsible use

- The recipes in this repo are yours to license. Pocket TTS code is MIT; the Pocket TTS **weights are CC-BY-4.0** and gated by Kyutai's use policy, which prohibits, among other things, **voice cloning or impersonation without explicit, lawful consent**, deception and harmful content. Models derived with this recipe inherit those terms: credit Kyutai and keep the restrictions in your model card.
- A model fine-tuned on one person's voice reproduces that voice: publish it only with their explicit consent.
- Check the license of your speech data before publishing a model. The datasets used here declare no license and are access-gated, and Bible recordings are often under publisher copyright.
