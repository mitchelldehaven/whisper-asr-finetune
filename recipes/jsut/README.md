# Recipe for JSUT corpus
Finetuning the Whisper ASR model using the [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut). 

## Step 0: Configuration
See `config.yaml` to check the configurations. Switch data.frontend to `None` if you prefer to use raw text instead of Japanese Kana text.

## Step 1: Download
Download the corpus data and unzip.
```
python download.py
```

## Step 2: Preprocess
Extract features from audio and text.
```
python preprocessing.py
```

## Step 3: Training
Finetune the Whisper ASR model. It takes less than 1 hour on XXX in Google Colab.
```
python train.py
```

## Step 4: Inference
Evaluate the finetuned model.
```
python inference.py
```
