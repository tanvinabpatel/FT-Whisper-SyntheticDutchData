#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script calculates the metrics such as Word Error Rate (WER) and Character Error Rate (CER) to quantify the accuracy of the model's transcriptions along with other measures. 
The output will include detailed error for each of the files and comparision with the reference text.

How to run:
	python3 scripts/evaluate.py --input path/to/the/decodefolder --ref reference_transcripts.txt

Note: Make sure the `InferModel` model location (checkpoint after fine-tuning) is updated in evaluate.py. This code supports .mp3 and .wav format

--input: 
	- path/to/the/decodefolder [Description: The path to the folder to be decoded.]
--ref: 
	- path/to/the/reference_transcripts.txt [Description: The path to the reference file.]
	- Reference file Format: [column1 column2] [filename.wav TranscriptionofFilename.wav]

Output: 
	- Transcriptions file `transcription.txt` is saved in the same directory as `path/to/the/decodefolder`
	- Performance of the system on the folder to be decoded is saved as `result.txt` in the `path/to/the/decodefolder`.

"""

import os
import torch
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import jiwer   # Using `jiwer` for WER computation
import soundfile as sf

def transcribe_audio_file(file_path, model, processor, device):
    # Read audio file
    audio_array, fs = sf.read(file_path)
    
    # Process the audio sample
    sampling_rate = processor.feature_extractor.sampling_rate
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
    inputs_features = inputs.input_features.to(device)
    
    # Forced decoder ids for transcription task, Generate predictions and Decode 
    forced_decoder_ids = processor.get_decoder_prompt_ids(task="transcribe")
    predicted_ids = model.generate(inputs_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def transcribe_folder(folder_path, model, processor, device):
    results = {}
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav') or f.endswith('.mp3')]
    
    file_out = open(f"{folder_path}/transcription.txt", "w")
    for file_name in audio_files:
        file_path = os.path.join(folder_path, file_name)
        #print(file_path)
        transcription = transcribe_audio_file(file_path, model, processor, device)
        results[file_name] = transcription    
    
        file_out.write(f"{file_name}\t{transcription}\n")
   
    file_out.close
    
    #print(type(results))
    return results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--ref', type=str, required=True)
    args = parser.parse_args()

    #Dataset to decode and reference file
    folder_path = args.input #"/home/tanvina/PrepINt/practical-assess-round2/DutchData/Testfolder"
    reference_file = args.ref #"/home/tanvina/PrepINt/practical-assess-round2/DutchData/DutchData_Syn/validation_transcriptions.txt"

    # Models location and # Load the processor and model
    InferModel = "/home/tanvina/PrepINt/practical-assess-round2/models/trainclean_nospkpunc/checkpoint-500"
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained(InferModel)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = transcribe_folder(folder_path, model, processor, device)
    
    #Get results using Jiwer
    references = {}
    with open(reference_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                file_name, transcription = parts
                references[file_name] = transcription
    
    # Match predictions and references
    pred_texts = []
    ref_texts = []
    for file_name, pred in predictions.items():
        if file_name in references:
            pred_texts.append(pred)
            ref_texts.append(references[file_name])
    
    #print(pred_texts);     print(ref_texts)
    # Compute WER, CER and get WER stats
    sent_wer = [];
    for i in range(len(ref_texts)):
    	#print(i, pred_texts[i], ref_texts[i])
    	sent_wer.append(jiwer.wer(ref_texts[i], pred_texts[i]))
    	#print(f"{pred_texts[i]}\n {ref_texts[i]} \n {sent_wer[i]}")
    
    finalout = jiwer.process_words(ref_texts, pred_texts)
    avg_wer = jiwer.wer(ref_texts, pred_texts)
    avg_cer = jiwer.cer(ref_texts, pred_texts)
    
    # Open the result file for writing the results of jiwer
    result_path = os.path.join(folder_path, "result.txt")
    with open(result_path, 'w') as fresult:
    # Write the alignment visualization
        alignment = jiwer.visualize_alignment(finalout)
        fresult.write(alignment)
        fresult.write("\n")
    
        # Write the WER and CER results
        fresult.write(f"Average WER: {avg_wer*100}\n")
        fresult.write(f"Average CER: {avg_cer*100}\n")
    
    # Optionally, print the results to the terminal
    print("Results","-" * 50)
    print(f"Average WER: {avg_wer*100} | Average CER: {avg_cer*100}")
    
#python3 scripts/evaluate.py --input /home/tanvina/PrepINt/FT-tiny-Dutch/samples/batch/ --ref /home/tanvina/PrepINt/FT-tiny-Dutch/samples/batch/batchreference.txt 

