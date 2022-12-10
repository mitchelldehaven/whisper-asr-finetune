from pathlib import Path
from whisper.normalizers.basic import BasicTextNormalizer
import argparse
from process_common_voice import parse_args

LANG_TO_TEXT_FILE = {
    "uk": "ubercorpus.tokenized.shuffled.txt"
}

if __name__ == "__main__":
    args = parse_args()
    normalizer = BasicTextNormalizer()
    text_file = LANG_TO_TEXT_FILE[args.lang_id]
    with (
        open(args.data_dir / "text" / args.lang_id / text_file) as source, 
        open(args.output_dir / args.lang_id / "train.txt", "w") as dest
    ):
        for line in source:
            normalized_text = normalizer(line.strip())
            print(normalized_text, file=dest)

