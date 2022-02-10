# Install numba ARM64
sudo apt install llvm-9
sudo ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
python3 -m pip install -U pip
python3 -m pip install --user -U llvmlite
sudo mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.h.bak
python3 -m pip install --user -U numba==0.50

# Update scipy para numba (could take several minutes to complete)
sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev gfortran
sudo -H python3 -m pip install -U scipy==1.1.0