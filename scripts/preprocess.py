#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script processes the database and prepares it for use by the training script. 
Since the data will be used to fine-tune the Whisper Model, the audio files and text need to meet the model’s requirements. 

How to run: 
	python3 scripts/preprocess.py --clean True (to clean the text data of symbols)
	python3 scripts/preprocess.py --clean False (to keep the text data unchanged)
	
--clean: [True or False]
	- If set to `True`, removes punctuation and special characters from the text data before processing. If set to `False`, the text data remains unchanged.
	- Default: `False`
    
Output:
	- Processed Audio: he wavefile parameters are altered to match the requiremnets of the Whisper-model.
	- --clean == `True`: The processed dataset will be saved as `processed_dataset_clean` in the `data` directory.
	- --clean == `False`: The processed dataset will be saved as `processed_dataset` in the `data` directory.
    
"""

import argparse
import re
from datasets import load_dataset, Audio
from transformers import WhisperProcessor

def prepare_dataset(example, processor, clean):
    audio = example["audio"]
    text = example["text"]
    
    if clean == True:
        # Remove punctuation if cleaning is required
        text = re.sub(r'[^\w\s]', '', text)
        #print(text)
    else:
        pass
    
    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=text,
        )

    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return example

def main(clean):
    # Load dataset
    print(clean) #Checking args 
    dataset = load_dataset("mariatepei/synthetic_accented_Dutch")

    # Load Model, processor and set the sampling frequency
    model = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model)
    sampling_rate = processor.feature_extractor.sampling_rate

    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Prepare dataset
    dataset = dataset.map(
        lambda example: prepare_dataset(example, processor, clean),
        remove_columns=["audio", "text"],
        num_proc=1
    )

    # Save processed dataset
    if clean:
        print('Processed Data: Cleaned text: Saving the dataset in data/processed_dataset_clean')
        print("-" * 50)
        output_dir = "data/processed_dataset_clean" 
    else:
        print('Processed Data: Saving the dataset in data/processed_dataset')
        print("-" * 50)
        output_dir = "data/processed_dataset"
        
    dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Text dataset based on option")
    parser.add_argument('--clean', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to clean the text by removing punctuation')
    #parser.add_argument('--output_dir', type=str, default="data/processed_dataset", help='Output directory to save the processed dataset')

    args = parser.parse_args()
   
    main(args.clean)

