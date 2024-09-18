#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs inferencing on given audio file(s) using the fine-tuned model. It supports both single-file and batch (folder) processing modes and saves the transcriptions in a JSON file in the same location as the wavefile(s). 

How to run:
	python3 scripts/inference_model.py --input /path/to/your/wave/filename.wav --batch 0
	python3 scripts/inference_model.py --input /path/to/your/wavefilesfolder/ --batch 1

Note: Make sure the `InferModel` model location (checkpoint after fine-tuning) is updated in `inference_model.py`. This code supports .mp3 and .wav format

--input: 
	- Path to the input folder or path to the file to be decoded.
--batch: 
	- Mode of processing. (0 for single-file processing, 1 for batch processing (all .wav files in the folder).)

Output: 
	- The transcriptions are saved as `transcription.json` in the same directory as `/path/to/your/wave/` or `/path/to/your/wavefilesfolder/`.

"""

import os
import argparse
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import json

def transcribe_audio_file(file_path, model, processor, device):
    # Read audio file
    audio_array, fs = sf.read(file_path)
    
    # Process the audio sample
    sampling_rate = processor.feature_extractor.sampling_rate
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    inputs_features = inputs.input_features.to(device)
    
    # Forced decoder ids for transcription task, # Generate predictions, # Decode the predictions
    forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
    predicted_ids = model.generate(inputs_features, forced_decoder_ids=forced_decoder_ids) # Ignore warning here
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def transcribe_folder(folder_path, model, processor, device):
    results = {}
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav') or f.endswith('.mp3')]
    
    for file_name in audio_files:
        file_path = os.path.join(folder_path, file_name)
        transcription = transcribe_audio_file(file_path, model, processor, device)
        results[file_name] = transcription
        print(f"File: {file_name}")
        print(f"Transcription: {transcription}")
        print("-" * 50)
    
    return results

def main(input_path, batch):
    if batch == 1:
        print("Processing folder...")
        results = transcribe_folder(input_path, model, processor, device)
        save_path = input_path
    else:
        print("Processing single file...")
        transcription = transcribe_audio_file(input_path, model, processor, device)
        save_path = os.path.dirname(input_path)
        results = {os.path.basename(input_path): transcription}
        print(f"File: {os.path.basename(input_path)}")
        print(f"Transcription: {transcription}")
        print("-" * 50)
        
    # Save results to JSON file
    output_file = f"{save_path}/""transcriptions.json"  # Save json in the location the wav was.
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    # Print transcriptions to the terminal    
    print(f"Transcriptions saved to {output_file}")  
    #print(json.dumps(results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Whisper model")
    parser.add_argument('--input', type=str, required=True, help='Path to input folder or file')
    parser.add_argument('--batch', type=int, required=True, choices=[0, 1], help='Batch mode (0 for single file, 1 for folder)')
    
    args = parser.parse_args()
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the processor and model
    InferModel = "/home/tanvina/PrepINt/practical-assess-round2/models/trainclean_nospkpunc/checkpoint-500"
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained(InferModel).to(device)
    
    main(args.input, args.batch)

#python3 scripts/inference_model.py --input /home/tanvina/PrepINt/FT-tiny-Dutch/samples/file/segment_143.mp3 --batch 0
#python3 scripts/inference_model.py --input /home/tanvina/PrepINt/FT-tiny-Dutch/samples/batch/ --batch 1

