This is the project for Berkeley [CS294/194-196 Large Language Model Agents | CS 194/294-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24)

Team Member:
Rachel Lin (3035629830)
Mutian Hong (3040824725)
Chuyan Zhou (3040814117)
Jan Chen (3040762169)
Lucy Struefing (3040858941)

tested on Ubuntu 22.02.
follow the following procedure to build up.
The complete architechure is built under `mutian-develop-v8` branch. The rag function in the main branch does not works well. So if you want the same architechure in the report, please switch to branch `mutian-develop-v8`.

### Backend
#### 1. install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### 2. install docker
```
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
newgrp docker
```

#### 3. setup opencv
```
sudo apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk2.0-dev \
    pkg-config
```

use tag to select ok

#### 5. setup conda env
```
conda create -n pdf2lec python=3.10
conda activate pdf2lec
pip install -r requirements.txt
sudo apt install ffmpeg
```

#### 6. create data folder
```
cd pdf2lec_llm/backend
mkdir -p data metadata
```

#### 7. start the backend
```
bash run.sh
```

### Frontend
```
cd ~/
cd pdf2lec_llm/frontend
sudo apt update
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

```
npm install
npm run dev
```