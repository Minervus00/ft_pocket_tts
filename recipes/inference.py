"""Synthesize speech with voice cloning using a Pocket TTS model folder
exported by recipes/export_release.py. Copy this script into the model folder.

    pip install pocket-tts scipy

    # one sentence
    python inference.py --voice reference.wav --text "Your sentence here."

    # one sentence per line of a UTF-8 text file -> out/0000.wav, out/0001.wav, ...
    python inference.py --voice reference.wav --text-file sentences.txt --output out/ --device cuda

--voice: 3-10 s of clean speech from the speaker to clone (any sample rate; the
output copies its acoustics, so a noisy prompt gives noisy speech). Only clone
voices you have permission to use.

Text must be written the way the training transcripts were (same orthography,
numbers as words). By default the config is model.yaml next to this script, which
loads the weights from the Hugging Face repo; pass --config model_local.yaml to
use local files (or the names you gave with export_release.py --config-name).
"""

import argparse
import time
import unicodedata
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from pocket_tts import TTSModel

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=str(Path(__file__).with_name("model.yaml")))
parser.add_argument("--voice", required=True, help="reference audio of the voice to clone")
text_in = parser.add_mutually_exclusive_group(required=True)
text_in.add_argument("--text", help="a single sentence to synthesize")
text_in.add_argument("--text-file", help="UTF-8 file, one sentence per line")
parser.add_argument("--output", default="output.wav", help="wav path for --text, directory for --text-file")
parser.add_argument("--device", default="cpu", help="cpu or cuda")
parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (default: config's 0.3)")
args = parser.parse_args()

if args.text is not None:
    sentences = [args.text]
else:
    sentences = [line.strip() for line in Path(args.text_file).read_text(encoding="utf-8").splitlines() if line.strip()]
# Training transcripts were NFC-normalized (recipes/prepare_hf_dataset.py).
sentences = [unicodedata.normalize("NFC", s) for s in sentences]

model = TTSModel.load_model(config=args.config, temp=args.temperature)
model.to(args.device)
voice_state = model.get_state_for_audio_prompt(args.voice)  # built once, reused for every sentence
sample_rate = model.config.mimi.sample_rate  # 24 kHz

if args.text is not None:
    out_paths = [Path(args.output)]
else:
    out_dir = Path(args.output)
    out_paths = [out_dir / f"{i:04d}.wav" for i in range(len(sentences))]

for text, out_path in zip(sentences, out_paths):
    start = time.perf_counter()
    audio = model.generate_audio(voice_state, text)  # 1-D float tensor
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio.detach().cpu().numpy(), -1.0, 1.0) * 32767).astype(np.int16)  # standard 16-bit wav
    scipy.io.wavfile.write(out_path, sample_rate, pcm16)
    seconds = audio.shape[-1] / sample_rate
    print(f"{out_path}  {seconds:.1f} s audio in {time.perf_counter() - start:.1f} s  | {text}")
