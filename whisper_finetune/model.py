import whisper
from pytorch_lightning import LightningModule
import evaluate
import torch
from torch import nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from .dataset import WhisperASRDataset, WhisperASRDataCollator

class WhisperModelModule(LightningModule):
    def __init__(self, config, model_name, lang, train_dataset=[], eval_dataset=[], epochs=0, steps_per_epoch=0) -> None:
        super().__init__()
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.model = whisper.load_model(model_name)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        # self.metrics_wer = evaluate.load("wer")
        # self.metrics_cer = evaluate.load("cer")

        self.config = config
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
    
    def forward(self, x):
        return self.model(x)


    def training_step_single_loader(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        # with torch.no_grad():
        audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        return loss
        # self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        # return loss


    def training_step_double_loader(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        out = self.model.decoder(dec_input_ids, None)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        return loss


    def training_step(self, batch, batch_idx):
        if type(batch) == list:
            audio_loss = self.training_step_single_loader(batch[0], batch_idx)
            text_loss = self.training_step_double_loader(batch[1], batch_idx)
            loss = audio_loss + text_loss
        else:
            loss = self.training_step_single_loader(batch, batch_idx)
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
    # def training_step(self, batch, batch_id):

    #     input_ids = batch["input_ids"]
    #     labels = batch["labels"].long()
    #     dec_input_ids = batch["dec_input_ids"].long()

    #     if input_ids.nelements():
    #         with torch.no_grad():
    #             audio_features = self.model.encoder(input_ids)

    #         out = self.model.decoder(dec_input_ids, audio_features)
    #     else:
    #         out = self.model.decoder(dec_input_ids, None)
    #     loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
    #     self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
    #     return loss
    
    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        # print(input_ids, flush=True)
        # print(input_ids.nelement(), flush=True)
        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True)

        return {
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
                        lr=self.config["learning_rate"], 
                        eps=self.config["adam_epsilon"]
                    )
        self.optimizer = optimizer

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            self.config["learning_rate"], 
            pct_start=0.1, 
            steps_per_epoch=self.steps_per_epoch, 
            epochs=self.epochs, 
            anneal_strategy="linear"
        )

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.config["warmup_steps"], 
        #     num_training_steps=self.t_total
        # )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
    
    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.config["batch_size"]))
                // self.config["gradient_accumulation_steps"]
                * float(self.config["num_train_epochs"])
            )
    
    # def train_dataloader(self):
    #     dataset = WhisperASRDataset(self.__train_dataset, self.tokenizer)
    #     return torch.utils.data.DataLoader(dataset, 
    #                 batch_size=self.config["batch_size"], 
    #                 drop_last=True, shuffle=True, num_workers=self.config["num_worker"],
    #                 collate_fn=WhisperASRDataCollator()
    #             )

    # def val_dataloader(self):
    #     dataset = WhisperASRDataset(self.__eval_dataset, self.tokenizer)
    #     return torch.utils.data.DataLoader(dataset, 
    #                 batch_size=self.config["batch_size"], 
    #                 num_workers=self.config["num_worker"],
    #                 collate_fn=WhisperASRDataCollator()
    #             )
    
