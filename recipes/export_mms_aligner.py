"""Export facebook/mms-1b-all with one language adapter merged in, as a plain
Wav2Vec2ForCTC folder that training/scripts/align_data.py can load via --model.

    uv run python ../recipes/moore/export_mms_aligner.py --lang mos --out ../data/models/mms-1b-mos
"""

import argparse
import json
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default="mos")
parser.add_argument("--out", required=True)
parser.add_argument("--base", default="facebook/mms-1b-all")
args = parser.parse_args()

model = Wav2Vec2ForCTC.from_pretrained(args.base, target_lang=args.lang, ignore_mismatched_sizes=True)

# The hub tokenizer holds one vocab per language; write this language's alone so
# the saved processor loads without needing target_lang.
full_vocab = json.loads(Path(hf_hub_download(args.base, "vocab.json")).read_text())
with tempfile.TemporaryDirectory() as tmp:
    vocab_path = Path(tmp) / "vocab.json"
    vocab_path.write_text(json.dumps(full_vocab[args.lang], ensure_ascii=False))
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.base)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

assert model.lm_head.out_features == len(tokenizer), (model.lm_head.out_features, len(tokenizer))
model.config.vocab_size = len(tokenizer)
model.save_pretrained(args.out)
processor.save_pretrained(args.out)
print(f"saved {args.lang} aligner to {args.out} (vocab {len(tokenizer)}): {''.join(sorted(k for k in tokenizer.get_vocab() if len(k) == 1))}")
