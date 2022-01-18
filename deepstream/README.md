# Deepstream install
Si necesita verificar su version de jetpack
sudo apt-cache show nvidia-jetpack

------------------------------------------
Instalar y probar deepstream examples
------------------------------------------

Referencia de instructivo para instalar deepstream (esta guia se basa en esto)
https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html

Instalar paqueteria gstreamer para Deepstream (con el docker deepstream ya vienen)
sudo apt-get update

sudo apt install \
libssl1.0.0 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4=2.11-1

a) Si se va a utilizar docker:
Dockers de Deepstream
https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html

b) Si se va a instalar deepstream nativo:
1-Descargar el paquete (requiere copiar el link en el explorador y logearse):
https://developer.nvidia.com/deepstream-6.0_6.0.0-1_arm64deb
2-Instalar
sudo apt-get install ./deepstream-6.0_6.0.0-1_arm64.deb


## Instalar numpy y opencv
sudo apt update
sudo apt install python3-numpy python3-opencv -y

Download Deepstream python examples:
cd /opt/nvidia/deepstream/deepstream/sources
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

## Instalar Python Bindings
Referencia:
https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html

Befor install gst-python install:
sudo apt-get install python3-dev libpython3-dev python-gi-dev -y

export GST_LIBS="-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0"
export GST_CFLAGS="-pthread -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include"

Download and install git repository of gst-python:
git clone https://github.com/GStreamer/gst-python.git
cd gst-python
git checkout 1a8f48a
git config --global http.sslverify false
./autogen.sh PYTHON=python3
./configure PYTHON=python3
make
sudo make install

*En caso de no encontrar el paquete pygobject instalarlo con apt-get
sudo apt-get install python-gi-dev

MetadaClass Access for Python
cd /opt/nvidia/deepstream/deepstream/lib
sudo python3 setup.py install

Run example
cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps/deepstream-test1
python3 deepstream_test_1.py /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264


# TRT PyCUDA gst-python install
------------------------------------------
Para ejecutar modelos con TRT directo (no deepstream)
Instalar gst-python y PyCUDA para Deepstream python
------------------------------------------
## Instalar numpy y opencv
sudo apt update
sudo apt install python3-numpy python3-opencv -y

Para ver las versiones de TRT instaladas:
dpkg -l | grep nvinfer

Instalar PyCUDA
Referencia
https://medium.com/dropout-analytics/pycuda-on-jetson-nano-7990decab299

Agregar referencia al PATH de CUDA (primero verificar que no funciona el comando $ nvcc --version)
sudo nano ~/.bashrc

Al final del archivo agregar
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

Cargar nueva configuracion
source ~/.bashrc

Verificar que ahora funciona el comando
nvcc --version

Instalar pip
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install cython
pip3 install pycuda


Demo de TensorRT para Jetson
https://github.com/jkjung-avt/tensorrt_demos

# Numba ARM64 install
------------------------------------------
Instalar numba en ARM64
------------------------------------------

## Instalar numba en ARM64
sudo apt install llvm-9
sudo ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
python3 -m pip install -U pip
python3 -m pip install --user -U llvmlite
sudo mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.h.bak
python3 -m pip install --user -U numba==0.50

## Actualizar scipy para numba (puede demorar bastante la instalación porque se compila scipy)
sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev gfortran
sudo -H python3 -m pip install -U scipy==1.1.0


