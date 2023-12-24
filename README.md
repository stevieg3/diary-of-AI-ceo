# diary-of-AI-ceo

Playing around with various generative AI tools to try and create a full-length (or close to) [Diary of a CEO](https://stevenbartlett.com/the-diary-of-a-ceo-podcast/) podcast.

High-level approach:
1. Get audio from existing podcasts on YouTube
1. Transcribe and apply speaker diarisation to audio
1. Fine-tune an LLM on transcriptions
1. Generate new podcasts with LLM
1. Use text-to-audio models to generate podcast

## Learnings
- Whisper (large-v2) is very good at transcription but not perfect. Also doesn't do diarisation out of the box.
- Diarisation is still very difficult. [whisperX](https://github.com/m-bain/whisperX) makes mistakes which require human correction e.g. identifying additional non-existent speakers. Couldn't find many alternatives to this package though.
- [Paperspace](https://www.paperspace.com/) is great for renting GPUs (and has very responsive customer service!).
- They offer machines with lots of pre-installed libraries ("ML-in-a-box") but this led to versioning issues when trying to run Hugging Face scripts. Opted to use vanilla Ubuntu OS and install CUDA drivers myself.
- Installing CUDA drivers is a pain! Best resource was this [video](https://youtu.be/ttxtV966jyQ?t=966&feature=shared).
- 16GB VRAM more than sufficient for transcribing.
- Training 7B LLM on long sequences (close to model context length) consumes more memory (VRAM). Needed 24GB GPU (P6000) to fine-tune a 7B model.
- Started with transformers/example script (no trainer) but moved to axototl (used by https://twitter.com/Teknium1) to see if it's easier to resume a peft run from a checkpoint

## Resources
[List and describe the resources used in this project, including any frameworks, libraries, or external tools. Provide links where appropriate.]