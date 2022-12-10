import yaml
from pathlib import Path 
import whisper
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[1]))
from whisper_finetune.dataset import load_data_list
from whisper_finetune.model import WhisperModelModule
from whisper_finetune.dataset import WhisperASRDataCollator, WhisperWavASRDataset, WhisperTextDataset
import math


def train():
    # load config 
    config_path = Path("config.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # dirs and paths
    in_data_dir = Path(config["path"]["preprocessed"])
    out_log_dir = Path(config["path"]["log"])
    checkpoint_dir = Path(config["path"]["checkpoint"])
    with_timestamps = bool(config["data"]["timestamps"])
    device = "gpu" if torch.cuda.is_available() else "cpu"

    out_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # tools
    whisper_options = whisper.DecodingOptions(
        language=config["data"]["lang"], without_timestamps=not with_timestamps
    )
    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True, language=config["data"]["lang"], task=whisper_options.task
    )

    # # list
    # train_list = load_data_list(in_data_dir / "train.txt")
    # val_list = load_data_list(in_data_dir / "val.txt")

    # logger
    tflogger = TensorBoardLogger(
        save_dir=out_log_dir,
        name=config["train_name"],
        version=config["train_id"]
    )

    # callback
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=checkpoint_dir / "checkpoint",
    #     filename="best_checkpoint",
    #     save_top_k=1 # all model save
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=checkpoint_dir / "checkpoint",
        filename="best_checkpoint",
        save_top_k=1,
        mode='min',
        verbose=True,
    )  
    callback_list = [
        checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]


    train_dataset = WhisperWavASRDataset(Path(config["data"]["lang"]) / "train.jsonl", whisper_tokenizer, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config["train"]["batch_size"], 
        drop_last=True, shuffle=True, num_workers=config["train"]["num_worker"],
        collate_fn=WhisperASRDataCollator()
    )

    train_text_dataset = WhisperTextDataset(Path(config["data"]["lang"]) / "train.txt", whisper_tokenizer)
    train_text_dataloader = torch.utils.data.DataLoader(
        train_text_dataset, 
        batch_size=config["train"]["batch_size"] * 2, 
        drop_last=True, shuffle=True, num_workers=1,
        collate_fn=WhisperASRDataCollator()
    )

    valid_dataset = WhisperWavASRDataset(Path(config["data"]["lang"]) / "dev.jsonl", whisper_tokenizer)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=config["train"]["batch_size"], 
        num_workers=config["train"]["num_worker"],
        collate_fn=WhisperASRDataCollator()
    )
    
    steps_per_epoch = math.ceil((len(train_dataset) / config["train"]["batch_size"]) / config["train"]["gradient_accumulation_steps"])
    model = WhisperModelModule(
        config["train"], config["model_name"], config["data"]["lang"], epochs=config["train"]["num_train_epochs"], steps_per_epoch=steps_per_epoch
    )

    trainer = Trainer(
        precision=16,
        accelerator=device,
        max_epochs=config["train"]["num_train_epochs"],
        accumulate_grad_batches=config["train"]["gradient_accumulation_steps"],
        logger=tflogger,
        callbacks=callback_list,
        devices=1,
        auto_select_gpus=True,
        multiple_trainloader_mode="min_size"
    )

    # trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.fit(model, train_dataloaders=[train_dataloader, train_text_dataloader], val_dataloaders=valid_dataloader)

if __name__ == "__main__":
    train()