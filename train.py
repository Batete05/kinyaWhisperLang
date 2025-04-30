#!/usr/bin/env python3
"""
train.py

Fine-tunes OpenAI's Whisper model for Kinyarwanda ASR on a custom dataset.
Requires: Hugging Face Transformers, torchaudio, and datasets.
"""

from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
import torchaudio
import os

from dataclasses import dataclass
import torch
from typing import Any, Dict, List, Union

import warnings


device = torch.device("cpu")  # instead of using MPS

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [f["labels"] for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            return_tensors="pt"
        )

        # Replace padding token id's of the labels by -100 so they are ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels
        return batch

# Load dataset from JSONL
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Load processor (tokenizer + feature extractor)
# Change "./kinya-whisper-model" to "openai/whisper-small" if training from scratch
processor = WhisperProcessor.from_pretrained("./kinya-whisper-model")
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# Preprocessing function
def prepare_example(batch):
    # Construct full path
    audio_path = os.path.join("Audio", os.path.basename(batch["Audio"]))
    # Load audio
    speech_array, sampling_rate = torchaudio.load(audio_path)
    batch["input_features"] = processor.feature_extractor(
        speech_array.squeeze().numpy(), sampling_rate=16000
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Map dataset
dataset = dataset.map(prepare_example)

# Load Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small") #first time
# model = WhisperForConditionalGeneration.from_pretrained("./kinya-whisper-model")

# Training arguments
training_args = TrainingArguments(
    output_dir="./kinya-whisper-model",
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    num_train_epochs=20,       # ⏱️ long-haul
    logging_steps=5,
    save_strategy="epoch",     # auto-saves checkpoints
    fp16=False,
    gradient_checkpointing=True,
    no_cuda=True
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer,  # <- still needed for saving and pushing to hub
    data_collator=data_collator     # <- the key fix!
)

warnings.filterwarnings("ignore", message=".*use_reentrant.*")

# Train the model
trainer.train()

# Save model
trainer.save_model("./kinya-whisper-model")

# Save pretrained
processor.save_pretrained("./kinya-whisper-model")









