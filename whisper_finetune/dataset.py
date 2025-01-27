
from email.mime import audio
import torch
from .frontend import TextFrontend
from .util import load_wav, load_txt
import whisper
import numpy as np
from typing import Union
from pathlib import Path
import json
import torchaudio
import random



def write_jsonl(json_list, json_file_path):
    with open(json_file_path, "w") as f:
        for sample in json_list:
            print(json.dumps(sample, ensure_ascii=False), file=f)


def read_jsonl(json_file_path):
    json_list = []
    with open(json_file_path) as f:
        for line in f:
            json_list.append(json.loads(line.strip()))
    return json_list


class WhisperASRDataset(torch.utils.data.Dataset):
    def __init__(
        self, id_mel_text_list: list,
        tokenizer: whisper.tokenizer,
        #sampling_rate: int=16000,
    ) -> None:
        super().__init__()

        assert len(id_mel_text_list) > 0
        #assert sampling_rate > 0

        self.id_mel_text_list = id_mel_text_list
        #self.sampling_rate = sampling_rate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.id_mel_text_list)

    def __getitem__(self, id):
        _, mel_path, text = self.id_mel_text_list[id]

        # text
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        # mel
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel.astype(np.float32))

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text
        }


class WhisperWavASRDataset(torch.utils.data.Dataset):
    def __init__(
        self, jsonl_file: list,
        tokenizer: whisper.tokenizer,
        training = False
    ) -> None:
        super().__init__()

        self.dataset = read_jsonl(jsonl_file)
        self.speed_range = list(range(88, 113))
        self.tokenizer = tokenizer
        self.training = training



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        waveform, sample_rate = torchaudio.load(sample["wav_path"])
        if self.training:
            speed = str(random.choice(self.speed_range) / 100)
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [["speed", speed]])
        waveform = whisper.pad_or_trim(waveform.flatten())
        mel = whisper.log_mel_spectrogram(waveform)
        text = sample["text"]

        # text
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text 
        }


class WhisperTextDataset():
    def __init__(self, text_file, tokenizer):
        self.dataset = []
        self.tokenizer = tokenizer
        with open(text_file) as f:
            for line in f:
                self.dataset.append(line.strip())
    

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        text = self.dataset[idx]
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)[:127]
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": None,
            "labels": labels,
            "dec_input_ids": text 
        }



class WhisperASRDataCollator():
    def __call__(self, features):
        input_ids, labels, dec_input_ids = [], [], []
        for feature in features:
            if feature["input_ids"] is not None:
                input_ids.append(feature["input_ids"])
            labels.append(feature["labels"])
            dec_input_ids.append(feature["dec_input_ids"])

        if input_ids:
            input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(label) for label in labels]
        dec_input_ids_length = [len(dec_input_id) for dec_input_id in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) 
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) 
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ] # 50257 is eot token id

        batch = {
            "input_ids": input_ids,
            "labels": torch.tensor(np.array(labels)),
            "dec_input_ids": torch.tensor(np.array(dec_input_ids))
        }

        # batch = {
        #     k: torch.tensor(np.array(v), requires_grad=False)
        #     for k, v in batch.items()
        # }
        # batch["input_ids"] = input_ids

        return batch


def valid_audio_text_safe(
        text, audio,
        text_max_length, audio_max_sample_length
    ):    
    if len(text) == 0:
        return False
    if len(text) > text_max_length:
        return False
    if audio is None:
        return False
    if len(audio) > audio_max_sample_length:
        return False
    return True

def save_data_list(
    data_list: list,
    list_path: Union[Path, str]
):
    with open(list_path, "w") as f:
        f.writelines("\t".join(x) + "\n" for x in data_list)


def load_data_list(
    list_path: Union[Path, str]
):
    return [
        x.strip("\n").split("\t")
        for x in open(list_path, "r").readlines()
    ]
