"""Push an exported Pocket TTS model folder (or an archive of one) to the Hugging Face Hub.

    python push_to_hub.py pocket-tts-moore-6l-step30000 --dry-run      # check what would be uploaded
    python push_to_hub.py pocket-tts-moore-6l-step30000                # private repo pocket-tts-moore-6l
    python push_to_hub.py pocket-tts-moore-step5000.tar.gz --repo-name pocket-tts-moore --public --gated auto

Reads HF_TOKEN and HF_USERNAME from .env next to this script (or the environment).

What it does:
  * accepts a folder produced by recipes/export_release.py, or a .zip / .tar / .tar.gz of one;
  * stages a copy in a temp dir (your folder is never modified): text files get
    REPLACE_WITH_HF_USER/<repo> placeholders replaced by <HF_USERNAME>/<repo-name>;
    large files are symlinked, not copied;
  * leaves out *_local.yaml (absolute local paths) and, unless --include-checkpoint,
    the *.pt training checkpoints (not needed for inference);
  * checks the model folder is complete and its configs point at the target repo;
  * creates the repo (private unless --public), optionally gates it, and uploads.

The repo name defaults to the one in the folder's hf:// config paths (e.g.
pocket-tts-moore-6l), so it matches what the model card documents.
"""

import argparse
import os
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

PLACEHOLDER_USER = "REPLACE_WITH_HF_USER"
PROJECT_ROOT = str(Path(__file__).resolve().parent)
TEXT_SUFFIXES = {".md", ".yaml", ".yml", ".py", ".json", ".txt"}
BINARY_TEXT_EXCEPTIONS = {"tokenizer.json"}  # large JSON copied verbatim


def load_env(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value and not os.environ.get(key.strip()):
            os.environ[key.strip()] = value


def unpack(source: Path, workdir: Path) -> Path:
    """Folder as-is; archives extracted into workdir. Returns the model folder."""
    if source.is_dir():
        return source
    name = source.name.lower()
    out = workdir / "extracted"
    if name.endswith(".zip"):
        with zipfile.ZipFile(source) as z:
            z.extractall(out)
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(source) as t:
            try:
                t.extractall(out, filter="data")  # safe extraction (Python >= 3.10.12)
            except TypeError:
                t.extractall(out)
    else:
        sys.exit(f"{source}: not a folder, .zip, .tar or .tar.gz")
    # An archive usually holds a single top-level folder.
    entries = [p for p in out.iterdir() if not p.name.startswith(".")]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else out


def repo_name_from_configs(folder: Path) -> str | None:
    for cfg in sorted(folder.glob("*.yaml")):
        m = re.search(r"hf://[^/\s]+/([^/\s]+)/model\.safetensors", cfg.read_text())
        if m:
            return m.group(1)
    return None


def stage(folder: Path, dest: Path, repo_id: str, include_checkpoint: bool) -> list[str]:
    skipped = []
    for src in sorted(folder.rglob("*")):
        rel = src.relative_to(folder)
        if src.is_dir() or any(part.startswith(".") for part in rel.parts):
            continue
        if src.name.endswith("_local.yaml") or (src.suffix == ".pt" and not include_checkpoint):
            skipped.append(str(rel))
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in TEXT_SUFFIXES and src.name not in BINARY_TEXT_EXCEPTIONS:
            text = src.read_text(encoding="utf-8")
            text = re.sub(rf"{PLACEHOLDER_USER}/[A-Za-z0-9._-]+", repo_id, text)
            # Absolute paths of this machine (e.g. in training_args.yaml) -> relative to the
            # pocket-tts checkout, as in the recipes; avoids publishing local user names.
            text = text.replace(f"{PROJECT_ROOT}/pocket-tts/", "").replace(f"{PROJECT_ROOT}/", "../")
            target.write_text(text, encoding="utf-8")
        else:
            target.symlink_to(src.resolve())
    return skipped


def check(staged: Path, repo_id: str) -> None:
    problems = []
    for required in ("model.safetensors", "tokenizer.json", "README.md"):
        if not (staged / required).exists():
            problems.append(f"missing {required}")
    configs = [p for p in staged.glob("*.yaml") if p.name != "training_args.yaml"]
    if not configs:
        problems.append("no model config (*.yaml)")
    for cfg in configs:
        text = cfg.read_text()
        for key in ("weights_path", "tokenizer_path"):
            m = re.search(rf"{key}:\s*(\S+)", text)
            if m and not m.group(1).startswith(f"hf://{repo_id}/"):
                problems.append(f"{cfg.name}: {key} is {m.group(1)}, expected hf://{repo_id}/...")
    for p in staged.rglob("*"):
        if p.is_file() and not p.is_symlink() and p.suffix in TEXT_SUFFIXES:
            text = p.read_text(errors="ignore")
            if PLACEHOLDER_USER in text:
                problems.append(f"{p.name}: still contains {PLACEHOLDER_USER}")
            if re.search(r"/home/[^/\s]+/", text):
                problems.append(f"{p.name}: contains an absolute /home/... path")
    if problems:
        sys.exit("refusing to upload:\n  - " + "\n  - ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="model folder, or .zip / .tar / .tar.gz of one")
    parser.add_argument("--repo-name", help="repo name without the user (default: from the folder's hf:// configs)")
    parser.add_argument("--public", action="store_true", help="create a public repo (default: private)")
    parser.add_argument("--gated", choices=["auto", "manual"], help="gate access to the repo (like kyutai/pocket-tts)")
    parser.add_argument("--include-checkpoint", action="store_true", help="also upload *.pt training checkpoints")
    parser.add_argument("--message", default=None, help="commit message")
    parser.add_argument("--dry-run", action="store_true", help="stage and check, but upload nothing")
    args = parser.parse_args()

    load_env(Path(__file__).with_name(".env"))
    token, user = os.environ.get("HF_TOKEN"), os.environ.get("HF_USERNAME")
    if not user:
        sys.exit("set HF_USERNAME in .env")
    if not token and not args.dry_run:
        sys.exit("set HF_TOKEN in .env (a token with write access)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        folder = unpack(Path(args.source).resolve(), tmp)
        repo_name = args.repo_name or repo_name_from_configs(folder)
        if not repo_name:
            sys.exit("could not infer the repo name from the configs: pass --repo-name")
        repo_id = f"{user}/{repo_name}"
        staged = tmp / "staged"
        skipped = stage(folder, staged, repo_id, args.include_checkpoint)
        check(staged, repo_id)

        files = sorted(p for p in staged.rglob("*") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        print(f"source : {folder}")
        print(f"repo   : https://huggingface.co/{repo_id} ({'public' if args.public else 'private'}"
              f"{', gated=' + args.gated if args.gated else ''})")
        print(f"upload : {len(files)} files, {total / 1e9:.2f} GB")
        for p in files:
            print(f"  {p.relative_to(staged)}  ({p.stat().st_size / 1e6:.1f} MB)")
        if skipped:
            print("skipped: " + ", ".join(skipped))
        if args.dry_run:
            print("dry run: nothing uploaded")
            return

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=True)
        if args.gated:
            api.update_repo_settings(repo_id, gated=args.gated)
        commit = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(staged),
            commit_message=args.message or f"Upload {Path(args.source).name}",
        )
        print(f"uploaded: {commit.commit_url if hasattr(commit, 'commit_url') else commit}")
        print(f"test it:  pocket-tts generate --config hf://{repo_id}/"
              f"{next(p.name for p in staged.glob('*.yaml') if p.name != 'training_args.yaml')} "
              f"--voice voice.wav --text '...'")


if __name__ == "__main__":
    main()
