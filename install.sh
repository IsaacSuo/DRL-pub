conda create -n tfdlenv python=3.8 -y
conda activate tfdlenv
pip install tensorflow

sudo apt update
sudo apt install pipx
pipx ensurepath

pipx install copier
copier --version
pip install gymnasium
pip install pyyaml
pip install matplotlib
pip install jupyter