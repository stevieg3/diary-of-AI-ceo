Instructions for setting up an environment in Terminal for running the transcription code.

Install miniconda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

Install dependencies:
```bash
conda create -n doaic python=3.10 -y
pip install pandas yt-dlp scrapetube python-dotenv
```

Install PyTorch version depending on CPU/GPU and CUDA driver:
```bash
Depending on compute/driver:
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch
```

Install whisperX:
```bash
pip install git+https://github.com/m-bain/whisperx.git
```