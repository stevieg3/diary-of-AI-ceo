Instructions for setting up CUDA drivers in terminal on GPU. Also installs PyTorch. Code is for CUDA drivers 12.1. Depending on GPU other versions may be needed instead.

Code taken from the excellent video [here](https://youtu.be/ttxtV966jyQ?t=966&feature=shared).

First install miniconda and create new env (used Python 3.10).

Install 12.1 CUDA drivers:
```bash
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
```

Install PyTorch:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Check PyTorch using GPU:
```python
import torch
torch.cuda.is_available()
```

Install other dependencies as required.
