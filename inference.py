"""
batch_inference.py

Performs transcription on all audio2 samples in a directory using a fine-tuned Whisper model.
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os

model_dir = "kinya-whisper-model"
audio_dir = "Audio"
output_file = "transcriptions.txt"

# Check if audio2 directory exists
if not os.path.exists(audio_dir):
    print(f"Error: Audio directory '{audio_dir}' not found.")
    exit(1)

print(f"🔍 Loading model from {model_dir}...")

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
processor = WhisperProcessor.from_pretrained(model_dir)

model.eval()
device = torch.device("cpu")

# Prepare the model's generation config (do this once)
model.generation_config.forced_decoder_ids = None

print(f"✅ Model loaded successfully!")
print(f"📂 Processing all audio2 files in {audio_dir}...")

# Open output file for writing
with open(output_file, "w") as f_out:
    # Get all WAV files in the directory
    audio_files = [f for f in sorted(os.listdir(audio_dir)) if f.lower().endswith(".wav")]

    if not audio_files:
        print(f"⚠️ No WAV files found in {audio_dir}")
        exit(0)

    print(f"🎵 Found {len(audio_files)} audio2 files to process")

    # Process each audio2 file
    for idx, filename in enumerate(audio_files):
        filepath = os.path.join(audio_dir, filename)
        print(f"[{idx + 1}/{len(audio_files)}] Transcribing {filename}...")

        try:
            # Load audio2
            waveform, sample_rate = torchaudio.load(filepath)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Process audio2
            input_features = processor.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features

            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    max_length=64,
                    num_beams=5,
                    do_sample=False,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                    length_penalty=1.2,
                    early_stopping=True,
                    task="transcribe"
                )

            # Decode the output
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Print and save the result
            print(f"📝 {filename}: {transcription}")
            f_out.write(f"{filename}: {transcription}\n")

        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"❌ {error_msg}")
            f_out.write(f"{filename}: ERROR - {str(e)}\n")

    print(f"✅ Processing complete! Results saved to {output_file}")






