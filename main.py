#!/usr/bin/env python3
"""
main.py

CLI: Records one audio sample, transcribes it, and gives an answer.
"""

import torch
import torchaudio
import sounddevice as sd
import pyttsx3
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# === Configuration ===
model_dir = "kinya-whisper-model"
record_duration = 5  # seconds
sample_rate = 16000
recorded_file = "recorded.wav"

# === Load model and processor ===
print("🔍 Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
processor = WhisperProcessor.from_pretrained(model_dir)
model.eval()
device = torch.device("cpu")

# === Load TTS engine ===
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# === Define QA Dictionary ===
qa_dict = {
    "abagabo": "Abagabo bubaka umuryango.",
    "amafaranga": "Amafaranga akoreshwa mu kugura ibikoresho.",
    "ikifuzo": "Ikifuzo ni icyo umuntu ashaka.",
    "imana": "Imana iturinda buri munsi.",
    "imyenda": "Imyenda igomba kuba isukuye.",
    "umupira": "Umupira ukinwa n’amakipe abiri.",
    "abana": "Abana bakenera urukundo n’uburere.",
    "ishuri": "Ishuri ni ahantu twigira ubumenyi.",
    "igiti": "Igiti gitanga umwuka mwiza.",
    "inzu": "Inzu iturwamo n’abantu.",
    "amaso": "Amaso adufasha kureba.",
    "amatwi": "Amatwi yumva amajwi.",
    "amazi": "Amazi ni ubuzima.",
    "ikirere": "Ikirere kirahindagurika buri munsi.",
    "umuryango": "Umuryango ni ishingiro ry’ubuzima.",
    "umuhanda": "Umuhanda ukoreshwa n’imodoka n’abantu.",
    "igitabo": "Igitabo kigira amagambo menshi yo gusoma.",
    "telefoni": "Telefoni ifasha abantu kuvugana.",
    "ikiruhuko": "Ikiruhuko ni igihe cyo kuruhuka no kurambura umubiri."
}

# === Record Audio ===
prompt_text = "Vuga ijambo mu Kinyarwanda, ndacyumva."
print(f"🎙️ {prompt_text}")
engine.say(prompt_text)
engine.runAndWait()

recording = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
torchaudio.save(recorded_file, torch.tensor(recording.T), sample_rate)
print("✅ Ijwi ribitswe!")

# === Load and preprocess audio ===
waveform, sr = torchaudio.load(recorded_file)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)

# === Transcribe ===
input_features = processor.feature_extractor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
).input_features

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

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
normalized = transcription.strip().lower()
print(f"📝 Transcription: {transcription}")

# === Respond ===
answer = qa_dict.get(normalized, "Mbabarira, sinasobanukiwe neza.")
print(f"🤖 Answer: {answer}")
engine.say(answer)
engine.runAndWait()
