🗣️ KinyaWhisperLang - Batch Audio Transcriber
KinyaWhisperLang is a powerful, lightweight batch transcription tool that leverages a fine-tuned version of OpenAI’s Whisper model to perform automatic speech recognition (ASR) on Kinyarwanda .wav audio files. Designed to process entire directories of audio at once, it's a practical solution for researchers, developers, and linguists working with low-resource languages.

🚀 What It Does
The included script batch_inference.py:

Loads a custom fine-tuned Whisper model for Kinyarwanda

Processes all .wav audio files in a specified directory

Automatically transcribes each file to text

Saves the transcriptions to a single output file

📂 Directory Structure
Your project folder should look like this:


KinyaWhisperLang/
├── Audio/                  # Folder containing your .wav audio files
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── kinya-whisper-model/    # Your local fine-tuned Whisper model folder
├── batch_inference.py      # Batch transcription script
└── transcriptions.txt      # Output file with transcriptions (auto-generated)
🧠 How It Works

model = WhisperForConditionalGeneration.from_pretrained("kinya-whisper-model")
processor = WhisperProcessor.from_pretrained("kinya-whisper-model")
It loops through all .wav files in the Audio/ folder, applies preprocessing (resampling, mono conversion), generates transcriptions using your Whisper model, and writes the results to transcriptions.txt.

🛠️ Requirements
Make sure you have the following Python packages installed:


pip install transformers torchaudio
▶️ Usage
Simply run:

python batch_inference.py
Make sure:

Your audio files are in the Audio/ directory

Your fine-tuned Whisper model is saved in a folder called kinya-whisper-model/

✨ Features
✅ Fully automatic transcription of audio folders

✅ Handles stereo/mono conversion and resampling to 16kHz

✅ Supports clean logging and error handling

✅ Saves results in a clean, structured .txt file

📌 Notes
This script is optimized for short, clear Kinyarwanda speech samples.

Ideal for batch processing small datasets or evaluating model performance quickly.

Not intended for real-time transcription or noisy long-form data.

📄 Example Output (transcriptions.txt)
sample1.wav: Mwaramutse neza, nitwa Ange.
sample2.wav: Abanyarwanda bakunda igihugu cyabo.
sample3.wav: ERROR - File corrupted or unreadable.
