import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from whisper.normalizers.basic import BasicTextNormalizer
import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[1]))
from whisper_finetune.dataset import read_jsonl, write_jsonl
import torchaudio

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--lang_id", type=str)
    parser.add_argument("--cv_dir_name", type=str, default="cv-corpus-11.0-2022-09-21")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    targz_file = args.cv_dir_name + "-" + args.lang_id
    # subprocess.call(f"tar xvf {args.data_dir / targz_file}.tar.gz -C {args.data_dir}", shell=True)
    # for mp3_file in (args.data_dir / args.cv_dir_name / args.lang_id / "clips").iterdir():
    #     subprocess.call(f"ffmpeg -hide_banner -loglevel error -i {mp3_file} -acodec pcm_s16le -ac 1 -ar 16000 {mp3_file}.wav", shell=True)
    normalizer = BasicTextNormalizer()
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)
    for dataset_file in ["train.tsv", "dev.tsv", "test.tsv"]:
        tsv_path = args.data_dir / args.cv_dir_name / args.lang_id / dataset_file
        skip_first = True
        dataset = []
        with open(tsv_path) as f:
            for line in f:
                if skip_first:
                    skip_first = False
                    continue
                split_line = line.strip().split("\t")
                mp3_filename = split_line[1]
                transcript = split_line[2]
                utt_id = Path(mp3_filename).name
                sample = {
                    "utt_id": utt_id,
                    "wav_path": str(args.data_dir.resolve() / args.cv_dir_name / args.lang_id / "clips" / (mp3_filename + ".wav")),
                    "text": normalizer(transcript).strip()
                }
                dataset.append(sample)
        (output_dir / args.lang_id).mkdir(exist_ok=True)
        write_jsonl(dataset, output_dir / args.lang_id / (dataset_file[:-4] + ".jsonl"))
        # with open(output_dir / args.lang_id / (dataset_file[:-4] + ".jsonl"), "w") as f:
        #     for sample in
        #     json.dump(dataset, f, indent=2, ensure_ascii=False)
        #     f.write("\n")
    