import yaml
from pathlib import Path 
import whisper
import torch
from tqdm import tqdm
import evaluate
from torch import autocast
from pytorch_lightning import Trainer

import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
from whisper_finetune.dataset import WhisperWavASRDataset, load_data_list, WhisperASRDataCollator
from whisper_finetune.model import WhisperModelModule

def inference():
    # load config 
    config_path = Path("config.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # dirs and paths
    in_data_dir = Path(config["path"]["preprocessed"])
    checkpoint_dir = Path(config["path"]["checkpoint"])
    with_timestamps = bool(config["data"]["timestamps"])
    #device = "gpu" if torch.cuda.is_available() else "cpu"

    # tools
    whisper_options = whisper.DecodingOptions(
        language=config["data"]["lang"], without_timestamps=not with_timestamps
    )
    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True, language=config["data"]["lang"], task=whisper_options.task
    )

    # list
    # test_list = load_data_list(in_data_dir / "test.txt")
    dataset = WhisperWavASRDataset(Path(".") / config["data"]["lang"] / "test.jsonl", whisper_tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16,
        collate_fn=WhisperASRDataCollator()
    )

    # load models
    epoch = config["inference"]["epoch_index"]
    # checkpoint_path = Path("checkpoint") / "checkpoint" / f"checkpoint-epoch=0003-v1.ckpt"
    checkpoint_path = Path("checkpoint") / "checkpoint" / f"best_checkpoint-v3.ckpt"
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict['state_dict']
    whisper_model = WhisperModelModule(
        config["train"], config["model_name"], config["data"]["lang"])
    whisper_model.load_state_dict(state_dict)

    # inference
    ref, hyp = [], []
    with autocast("cuda", dtype=torch.float16):
        for b in tqdm(loader):
            input_id = b["input_ids"].cuda()
            label = b["labels"].long().cuda()
            with torch.no_grad():
                hypothesis = whisper_model.model.decode(input_id, whisper_options)
                for h in hypothesis:
                    hyp.append(h.text)
                
                for l in label:
                    l[l == -100] = whisper_tokenizer.eot
                    r = whisper_tokenizer.decode(l, skip_special_tokens=True)
                    ref.append(r)

    for r, h in zip(ref, hyp):
        print("-"*10)
        print(f"reference:  {r}")
        print(f"hypothesis: {h}")

    # compute CER
    cer_metrics = evaluate.load("cer")
    cer = cer_metrics.compute(references=ref, predictions=hyp)
    print(f"CER: {cer}")

if __name__ == "__main__":
    inference()