🗣️ KinyaWhisperLang
KinyaWhisperLang is a personalized adaptation of OpenAI’s Whisper model, fine-tuned for Kinyarwanda automatic speech recognition (ASR). This project is aimed at contributing to accessible voice AI for underrepresented languages like Kinyarwanda, and serves as a baseline for future experimentation and research.

🤗 Hugging Face Model
You can host your own fine-tuned model on Hugging Face and use it like this:

python
Copy
Edit
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load fine-tuned KinyaWhisperLang model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("your-username/kinyaWhisperLang")
processor = WhisperProcessor.from_pretrained("your-username/kinyaWhisperLang")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("your_audio.wav")
inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

# Generate prediction
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("🗣️ Transcription:", transcription)
🏋️ Training Details
Model: openai/whisper-small

Epochs: 80

Batch size: 4

Learning rate: 1e-5

Optimizer: Adam

Final loss: 0.00024

WER (Word Error Rate): 51.85%

⚠️ Limitations
This is an early-stage prototype trained on a small dataset (102 samples). It performs best on short, clean Kinyarwanda audio. It may struggle with background noise, long-form speech, or domain-specific vocabulary. It is not yet ready for production use.

📚 Citation
If you use this project or model, please cite:

bibtex
Copy
Edit
@misc{batete2025kinyawhisperlang,
  author       = {Batete},
  title        = {KinyaWhisperLang: Fine-Tuning Whisper for Kinyarwanda ASR},
  year         = {2025},
  howpublished = {\url{https://github.com/Batete05/kinyaWhisperLang}},
  note         = {Version 1.0}
}
📬 Contact
Maintained by Batete (Batete05)
✉️ bateteangenadette@gmail.com
🔗 https://github.com/Batete05
