#!/bin/bash
# Dyula Pocket TTS pipeline. Every stage is idempotent: finished outputs are skipped.
#
#   bash recipes/dyula/run_dyula.sh prep          # HF dataset -> wavs + manifests + tokenizer
#   bash recipes/dyula/run_dyula.sh align         # MMS forced alignment of all splits
#   bash recipes/dyula/run_dyula.sh check         # verify access to the gated kyutai/pocket-tts weights
#   bash recipes/dyula/run_dyula.sh train fr24l   # fine-tune the 24-layer teacher from french_24l
#   bash recipes/dyula/run_dyula.sh distill [variant]   # 24L teacher -> 6L student (default variant fr6l; fem_fr6l = female voice)
#   bash recipes/dyula/run_dyula.sh export <run> <step> <out_dir> [hf_repo_id]   # package a checkpoint
#   bash recipes/dyula/run_dyula.sh tts <run> <checkpoint.pt> <voice.wav> "<text>" [out.wav]
#
# <run> is a folder name under runs/: dyula_fr24l or dyula_fr6l_distill.
set -euo pipefail

# ---- language-specific settings ---------------------------------------------
LANG_ID=dyula                              # names data/<LANG_ID>, runs/<LANG_ID>_*, config files
HF_DATASET=Minervus00/dyula-speech-bible   # gated: HF_TOKEN is read from the project .env; 16 kHz audio, resampled to 24 kHz
TEXT_COL=transcription
SPEAKER_COL=speaker_id
GROUP_COL=chapter                          # chapter/recording; consecutive clips of a group get merged
HOLDOUT_SPEAKERS=SPEAKER_03                # kept out of training to test cloning of unseen voices
MMS_LANG=dyu                               # MMS-1b-all adapter used for forced alignment (ISO 639-3)
# ------------------------------------------------------------------------------

RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$RECIPE/../.." && pwd)
REPO=$ROOT/pocket-tts
DATA=$ROOT/data/$LANG_ID
ALIGNER=$ROOT/data/models/mms-1b-$MMS_LANG
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
# Gated datasets/models: export HF_TOKEN (and HF_USERNAME) from the project .env if present.
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi

refuse_if_running() {  # never start a second process on the same config/run dir
  if pgrep -f "python -m training.train .*$(basename "$1")" > /dev/null; then
    echo "$(basename "$1") is already training (pgrep -af training.train); not starting a second copy"; exit 1
  fi
}

CALLER_DIR=$PWD  # user-given output paths are relative to where the script was called from
cd "$REPO"  # configs use paths relative to the pocket-tts checkout
stage=${1:-}
shift || true

case "$stage" in
  prep)
    [ -f "$DATA/train.jsonl" ] || $UV run python "$ROOT/recipes/prepare_hf_dataset.py" \
      --dataset "$HF_DATASET" --text-col "$TEXT_COL" --speaker-col "$SPEAKER_COL" --group-col "$GROUP_COL" \
      --holdout-speakers "$HOLDOUT_SPEAKERS" --out "$DATA" --workers 8
    [ -f "$DATA/tokenizer.model" ] || $UV run python -m training.scripts.train_tokenizer \
      "$DATA/tokenizer" "$DATA/train.jsonl" --character-coverage 1.0
    [ -f "$DATA/tokenizer.json" ] || $UV run python -m training.scripts.convert_tokenizer \
      "$DATA/tokenizer.model" --out "$DATA/tokenizer.json" --texts "$DATA/valid.jsonl"
    ;;
  align)
    [ -d "$ALIGNER" ] || $UV run python "$ROOT/recipes/export_mms_aligner.py" --lang "$MMS_LANG" --out "$ALIGNER"
    for split in valid unseen train; do
      [ -s "$DATA/$split.jsonl" ] || continue
      # --resume appends to a partial output, so an interrupted run continues where it stopped.
      $UV run python -m training.scripts.align_data "$DATA/$split.jsonl" "$DATA/${split}_aligned.jsonl" \
        --model "$ALIGNER" --batch-size 16 --resume
    done
    wc -l "$DATA"/*_aligned.jsonl
    ;;
  check)
    $UV run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('kyutai/pocket-tts', 'languages/french_24l/model.safetensors', revision='d0ebc5deb50bc6b225f686782f124dd37e2dca0a')
print('OK: gated kyutai/pocket-tts weights are accessible')"
    ;;
  train)
    variant=${1:?usage: train fr24l}
    cfg=$RECIPE/finetune_${LANG_ID}_$variant.yaml
    refuse_if_running "$cfg"
    mkdir -p "$ROOT/runs"
    # Auto-resumes from the newest checkpoint in the run dir if one exists.
    nohup $UV run python -m training.train "$cfg" > "$ROOT/runs/${LANG_ID}_$variant.out" 2>&1 &
    echo "started PID $! -> tail -f $ROOT/runs/${LANG_ID}_$variant.out"
    ;;
  distill)
    variant=${1:-fr6l}
    cfg=$RECIPE/distill_${LANG_ID}_${variant}.yaml
    refuse_if_running "$cfg"
    teacher=$(grep -E "^distill_teacher_weights:" "$cfg" | awk '{print $2}')
    [ -f "$teacher" ] || { echo "teacher checkpoint not found: $teacher -- edit distill_teacher_weights in $cfg"; exit 1; }
    if pgrep -f "python -m training.train .*finetune_${LANG_ID}" > /dev/null; then
      echo "a teacher run is still training; stop it first (the GPU cannot hold both)"; exit 1
    fi
    mkdir -p "$ROOT/runs"
    nohup $UV run python -m training.train "$cfg" > "$ROOT/runs/${LANG_ID}_${variant}_distill.out" 2>&1 &
    echo "started PID $! (teacher: $teacher) -> tail -f $ROOT/runs/${LANG_ID}_${variant}_distill.out"
    ;;
  export)
    run=${1:?run name}; step=${2:?step}; out=$(cd "$CALLER_DIR" && realpath -m "${3:?output dir}")
    infer_cfg=$RECIPE/${LANG_ID}_24l_infer.yaml; default_repo=REPLACE_WITH_HF_USER/pocket-tts-$LANG_ID
    [[ $run == *6l* ]] && { infer_cfg=$RECIPE/${LANG_ID}_6l_infer.yaml; default_repo=$default_repo-6l; }
    repo=${4:-$default_repo}
    # On CPU, so a running training job keeps the GPU.
    CUDA_VISIBLE_DEVICES="" $UV run python "$ROOT/recipes/export_release.py" --run-dir "$ROOT/runs/$run" \
      --step "$step" --out "$out" --repo-id "$repo" --infer-config "$infer_cfg" --config-name "$LANG_ID" \
      --with-training-checkpoint
    sed "s/model\.yaml/${LANG_ID}.yaml/; s/model_local\.yaml/${LANG_ID}_local.yaml/" \
      "$ROOT/recipes/inference.py" > "$out/inference.py"
    echo "exported to $out -- add a README.md (model card) and samples/ before uploading"
    ;;
  tts)
    run=${1:?run name, e.g. ${LANG_ID}_fr24l}; ckpt=${2:?checkpoint .pt}; voice=${3:?voice wav}; text=${4:?text}
    out=${5:-$ROOT/runs/$run/tts_$(date +%H%M%S).wav}
    infer_cfg=$RECIPE/${LANG_ID}_24l_infer.yaml
    [[ $run == *6l* ]] && infer_cfg=$RECIPE/${LANG_ID}_6l_infer.yaml  # distilled student
    $UV run pocket-tts generate --config "$infer_cfg" --checkpoint "$ckpt" \
      --voice "$voice" --text "$text" --output-path "$out" --device cuda
    echo "wrote $out"
    ;;
  *)
    sed -n 2,12p "$RECIPE/$(basename "${BASH_SOURCE[0]}")"; exit 1 ;;
esac
