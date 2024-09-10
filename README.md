# Attendi Task: Whisper Fine-Tuning on Synthetic Dutch Database

This codebase is to fine-tune a Dutch Accented Database on the whisper-tiny model

## Setup
**Pre-requisites**

```
ProjectDirectoryName
├── Readme.md
├── requirements.txt
└── scripts
    ├── evaluate.py
    ├── inference_model.py
    ├── preprocess_punc.py
    ├── preprocess.py
    ├── train.py
    └── train_spkpunc.py
```

**Create virtual virtual environment and activate:**

```	
	python3 -m venv .venv
	source .venv/bin/activate  # On Linux
	
	pip install --upgrade pip
```

**Install dependencies:**

```sh
	pip3 install -r requirements.txt
```

## Database Details

**Database:** Accented Dutch speech dataset generated synthetically using a speech synthesis model

**Details:**

Location: https://huggingface.co/datasets/mariatepei/synthetic_accented_Dutch

```
| Dataset Split | Number of Files | Total Duration (hours) |
|---------------|-----------------|------------------------|
| Train         | 1466            | 10.31                  |
| Validation    | 357             | 2.5                    |
```
The database need not be downloaded, it will be directly processed and saved by the scripts.

## Training and Evaluation
Training: Fine-tuning on the Whisper-tiny model.
Location: https://huggingface.co/openai/whisper-tiny

The model need not be downloaded, it will be directly processed and used by the scripts.

### Part A: Without Spoken Punctuations
In this case, the spoken audio is delivered in a standard manner without any spoken punctuation. The corresponding text transcripts contain written punctuation marks in symbols such as ,/./?/! (i.e., periods, commas, and question marks).

**Processing the Dataset:**

This script processes the database and prepares it for use by the training script. 
Since the data will be used to fine-tune the Whisper Model, the audio files and text need to meet the model's requirements.

```sh
	python3 scripts/preprocess.py --clean False 
```

- **Arguments:**
  - **`--clean`**:
    - **Description:** If set to `True`, removes punctuation and special characters from the text data before processing. If set to `False`, the text data remains unchanged.
    - **Default:** `False`
- **Output:**
    - **`Processed Audio`:** The wavefile parameters are altered to match the requiremnets of the Whisper-model
    - **`--clean` == `True`:** The processed dataset will be saved as `processed_dataset_clean` in the `data` directory.
    - **`--clean` == `False`:** The processed dataset will be saved as `processed_dataset` in the `data` directory.


**Training the Model:** 

This script uses either the `processed dataset` or `processed_dataset_clean` database to fine-tune a model based on Whisper-tiny.

```sh
	python3 scripts/train.py 
```
**Note:** This scripts can use GPU/CPU for training. Set the parameters, modelname and other parameters in the script accordingly.

- **Output:**
    - **Model Checkpoints:** The trained models are saved in the `model` directory
    
### Part B: With Spoken Punctuations
In this case, the audio files contain spoken punctuation. E.g., "Hallo komma hoe gaat het vraagteken Ik ben goed punt"


**Processing the Dataset:**

This script processes the database and prepares it for use by the training script. 
Since the data will be used to fine-tune the Whisper Model, the audio files and text need to meet the model's requirements. The text transcripts, if containing punctuation symbols (,/./?/!) should be mapped to tokens (`<punt>, <komma>, <vraagteken>, <uiteoepteken>`).

```bash
    python3 scripts/preprocess_punc.py 
```

- **Output:**
    - **`Processed Audio`:** The wavefile parameters are altered to match the requiremnets of the Whisper-model
    - **`Processed Text`:** The punctuation symbols are converted into tokens in the text.
    - The processed dataset will be saved as `processed_dataset_spkpunc` in the `data` directory.

**Training the Model:**

This script uses the `processed_dataset_spkpunc` database to fine-tune a model based on Whisper-tiny.

```sh
    python3 scripts/train_spkpunc.py 
```
**Note:** This scripts can use GPU/CPU for training. Set the parameters, modelname and other parameters in the script accordingly.

- **Output:**
    - **Model Checkpoints:** The trained models are saved in the `model` directory
    
**Evaluation:**

The script calculates the metrics such as Word Error Rate (WER) and Character Error Rate (CER) to quantify the accuracy of the model's transcriptions along with other measures. The output will include detailed error for each of the files and comparision with the reference text.

```sh
    python3 scripts/evaluate.py --input path/to/the/decodefolder --ref reference_transcripts.txt
```
**Note:** Make sure the `InferModel` model location (checkpoint after fine-tuning) is updated in `evaluate.py`. This code supports .mp3 and .wav format

- **Arguments:**
  - **`--input`:** `path/to/the/decodefolder`
    - **Description:** The path to the folder to be decoded.
  
  - **`--ref`:** `path/to/the/reference_transcripts.txt`
    - **Description:** The path to the reference file.
    - **Format:** 	`[column1	column2]`
    			`[filename.wav TranscriptionofFilename.wav]`
  
- **Output:**
  - Transcriptions file `transcription.txt` is saved in the same directory as `path/to/the/decodefolder` 
  - Performance of the system on the folder to be decoded is saved as `result.txt` in the `path/to/the/decodefolder` .
 

## Inferencing
This script performs inferencing on given audio file(s) using the fine-tuned model. It supports both single-file and batch (folder) processing modes and saves the transcriptions in a JSON file in the same location as the wavefile(s). 

```sh
    python3 scripts/inference_model.py --input /path/to/your/wave/filename.wav --batch 0
    python3 scripts/inference_model.py --input path/to/your/wavefilesfolder/ --batch 1
```
**Note:** Make sure the `InferModel` model location (checkpoint after fine-tuning) is updated in `inference_model.py`. This code supports .mp3 and .wav format

- **Arguments:** 
  - **`--input`:** Path to the input folder or path to the file to be decoded.
  - **`--batch`:** Mode of processing. (`0` for single-file processing, `1` for batch processing (all .wav files in the folder).)

- **Output:**
  - The transcriptions are saved as `transcription.json` in the same directory as `/path/to/your/wave` or `/path/to/your/wavefilesfolder/` 
  
Refer to end of script for how to run using the sample files provided in the `samples` directory.
 
## Results
**Evaluation Datasets:**

_SynDutch-Train_: Subset of the training dataset from the Synthetic Accented Dutch database: 100 files

_SynDutch-Valid_: Subset of the validation dataset from the Synthetic Accented Dutch database: All 357 files

_CGN-read_: Read speech from the CGN Corpus (Out-of-Domain): 13 files


```
|        Model                |             Results: % WER (% CER)                |
|-----------------------------|---------------------------------------------------|
|        Description          | SynDutch-Train | SynDutch-Valid |     CGN-Read    |
|-----------------------------|----------------|----------------|-----------------|
| train_nospkpunc-500         | 42.75 (14.46)  | 57.00 (19.42)  | 335.25 (196.18) |
| trainclean_nospkpunc-500    | 35.38 (15.91)  | 42.44 (12.38)  | 78.84 (29.51)   |
| train_spkpunc-500           | 90.82 (83.99)  | 88.26 (83.34)  | 181.41 (165.76) |

```

