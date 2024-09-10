#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script uses either the `processed_dataset` or `processed_dataset_clean` database to fine-tune a model based on Whisper-tiny.

How to run:
	python3 scripts/train.py 

Note: This scripts can use GPU/CPU for training. Set the parameters, modelname and other parameters in the script accordingly.

Output:
	- Model Checkpoints: The trained models are saved in the `model` directory

"""

import os
import torch
from datasets import load_from_disk, load_metric
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Uncomment if you do not have a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """ Define data collator to handle batching and padding of audio features and labels """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    #pred_logits = pred.predictions
    pred_ids = pred.predictions
    #pred_ids = pred_logits.argmax(-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # We need to truncate the predictions and labels
    #pred_str = [s.strip() for s in pred_str]
    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    #label_str = [s.strip() for s in label_str]
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
    
def load_subset(dataset, percentage):
    
    train_size = int(len(dataset["train"]) * percentage)
    validation_size = int(len(dataset["validation"]) * percentage)
    return {
        "train": dataset["train"].select(range(train_size)),
        "validation": dataset["validation"].select(range(validation_size))
    }


def main_train():
    # Load preprocessed dataset
    dataset = load_from_disk("data/processed_dataset_clean")
    percentage_use = 1.0 #Set this parameter < 1.0 if want to train with a subset of training data
    dataset = load_subset(dataset, percentage_use)
    
    
    # Uncomment if you do not have a GPU
    device = torch.device("cpu")
    model.to(device)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/trainclean_nospkpunc", #Choose any model name
        eval_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        logging_dir="./logs",
        do_predict=True,
        save_steps=100,
        eval_steps=50,
        max_steps=500,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        save_total_limit=2,
        predict_with_generate=True,
        learning_rate=1e-5,
        warmup_steps=500,
        #num_train_epochs=3,
    	gradient_accumulation_steps=16,  # Accumulate gradients over 16 steps
        gradient_checkpointing=True,
    	dataloader_num_workers=1  # Use multiple workers for data loading
    )

    # Initialize data collator and metric
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
    )

    # Start training
    trainer.train()
    
    # Evaluate the test (same as validation)
    #eval_results = trainer.evaluate()
    #print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    # Load model and processor
    modelname = "openai/whisper-tiny"
    model = WhisperForConditionalGeneration.from_pretrained(modelname)
    processor = WhisperProcessor.from_pretrained(modelname)
    metric = load_metric('wer')
    
    main_train()
    
