"""Package a training checkpoint as a pocket-tts model folder ready for Hugging Face.

Writes into --out:
  model.safetensors   flow_lm + mimi weights in the format the pocket-tts wheel loads
                      (raw weights by default -- what the run's samples/ were generated
                      with; --use-ema exports the EMA shadow instead)
  tokenizer.json      Mooré tokenizer the model was trained with (+ tokenizer.model, the sentencepiece source)
  <name>.yaml         model config pointing at the files above via hf://<repo-id>/...   (--config-name)
  <name>_local.yaml   same config with absolute local paths, for testing before upload
  training_args.yaml  resolved training config of the run
  checkpoint_<N>.pt   (with --with-training-checkpoint) raw training checkpoint, to resume or re-export

Run from the pocket-tts repo root:
    # 24-layer teacher
    uv run python ../recipes/export_release.py --run-dir ../runs/moore_fr24l --step 5000 \
        --out ../pocket-tts-moore-step5000 --repo-id <user>/pocket-tts-moore \
        --infer-config ../recipes/moore/moore_24l_infer.yaml --config-name moore
    # 6-layer distilled student
    uv run python ../recipes/export_release.py --run-dir ../runs/moore_fr6l_distill --step 50000 \
        --out ../pocket-tts-moore-6l-step50000 --repo-id <user>/pocket-tts-moore-6l \
        --infer-config ../recipes/moore/moore_6l_infer.yaml --config-name moore
"""

import argparse
import re
import shutil
from pathlib import Path

import yaml

from training.args import load_args
from training.checkpointing import EMA, export_pocket_safetensors, load_checkpoint
from training.modules.builders import build_models

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--repo-id", default="REPLACE_WITH_HF_USER/REPLACE_WITH_REPO_NAME")
parser.add_argument("--use-ema", action="store_true")
parser.add_argument("--with-training-checkpoint", action="store_true")
parser.add_argument(
    "--infer-config",
    required=True,
    help="inference config matching the run's architecture, e.g. recipes/moore/moore_24l_infer.yaml "
    "(teacher) or recipes/moore/moore_6l_infer.yaml (distilled student)",
)
parser.add_argument("--config-name", default="model", help="basename of the written configs: <name>.yaml, <name>_local.yaml")
args = parser.parse_args()

run_dir = Path(args.run_dir).resolve()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
ckpt = run_dir / f"checkpoint_{args.step:08d}.pt"
assert ckpt.exists(), f"{ckpt} not found (only the last few checkpoints are kept)"

train_args = load_args(run_dir / "args.yaml")
model, mimi, _ = build_models(train_args)
ema = EMA(model, train_args.ema_decay) if args.use_ema else None
step = load_checkpoint(ckpt, model, ema=ema)
assert step == args.step, (step, args.step)
export_pocket_safetensors(out / "model.safetensors", model.flow_lm, mimi, ema)

tokenizer_json = Path(train_args.model_overrides["flow_lm.lookup_table.tokenizer_path"])
shutil.copy2(tokenizer_json, out / "tokenizer.json")
if tokenizer_json.with_suffix(".model").exists():
    shutil.copy2(tokenizer_json.with_suffix(".model"), out / "tokenizer.model")
shutil.copy2(run_dir / "args.yaml", out / "training_args.yaml")
if args.with_training_checkpoint:
    shutil.copy2(ckpt, out / ckpt.name)

# Model config: the recipe's inference config (--infer-config: architecture, French text
# replacements removed) with weights and tokenizer pointed at this folder.
infer_cfg = Path(args.infer_config).read_text()
infer_cfg = "\n".join(line for line in infer_cfg.splitlines() if not line.startswith("#"))


def with_paths(text: str, weights: str, tokenizer: str) -> str:
    text = re.sub(r"(?m)^weights_path:.*$", f"weights_path: {weights}", text)
    text = re.sub(r"(?m)^weights_path_without_voice_cloning:.*\n", "", text)
    text = re.sub(r"(?m)^(\s*tokenizer_path:).*$", rf"\1 {tokenizer}", text)
    return text.strip() + "\n"


n_layers = len(model.flow_lm.transformer.layers)
header = f"# pocket-tts {n_layers}-layer model exported from {run_dir.name}, step {step}.\n"
(out / f"{args.config_name}.yaml").write_text(
    header + with_paths(infer_cfg, f"hf://{args.repo_id}/model.safetensors", f"hf://{args.repo_id}/tokenizer.json")
)
(out / f"{args.config_name}_local.yaml").write_text(
    header + with_paths(infer_cfg, str(out / "model.safetensors"), str(out / "tokenizer.json"))
)
for name in (f"{args.config_name}.yaml", f"{args.config_name}_local.yaml"):
    yaml.safe_load((out / name).read_text())  # fail loudly on a broken config
print(f"exported step {step} ({'EMA' if args.use_ema else 'raw'} weights, {n_layers} layers) to {out}")
