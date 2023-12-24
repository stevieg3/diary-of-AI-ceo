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


```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

conda create -n doaic python=3.10 -y
pip install pandas yt-dlp scrapetube python-dotenv

Depending on compute/driver:
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch

pip install git+https://github.com/m-bain/whisperx.git
```

Installing cuda drivers 12.1

Following this: https://www.youtube.com/watch?v=ttxtV966jyQ&list=PLBoQnSflObcnZktTLziVYvXg-8imDiLhy&index=3

First install miniconda
```
sudo apt-get install linux-headers-$(uname -r)

sudo apt-key del 7fa2af80

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin

sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb

sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update

sudo apt-get -y install cuda

export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
