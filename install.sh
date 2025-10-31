conda create -n tfdlenv python=3.10 -y
conda activate tfdlenv
pip install tensorflow

sudo apt update
sudo apt install pipx
pipx ensurepath

pipx install copier
copier --version
pip install gymnasium

pip install matplotlib
