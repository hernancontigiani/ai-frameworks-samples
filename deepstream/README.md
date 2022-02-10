# Deepstream install
![](https://developer.nvidia.com/sites/default/files/deepstreamsdk3-workflow.png)


# How to
## Download Jetpack and install
Reference:\
[Deepstream Jetpack from NVIDIA](https://developer.nvidia.com/embedded/jetpack)

In case you have already installed jetpack, you could verify the version with:
```sh
$ sudo apt-cache show nvidia-jetpack
```

## First of all, update repositories and install nano
```sh
$ sudo apt-get update
$ sud apt-get install nano -y
```

## Install Dreepstream dependencies
Reference:\
[Deepstream guide quickstart](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html)

Install gstreamer dependencies:
```sh
$ sudo apt install -y \
libssl1.0.0 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4=2.11-1
```

Update L4T repositories (for Jetpack 4.6 only):\
This fixs the warning "undefined symbol: NvBufSurfTransformAsync"
- Open l4t resources 
```sh
$ sudo nano /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
```
- Edit this for Jetson nano Jetpack 4.6
```
deb https://repo.download.nvidia.com/jetson/common r32.6 main
deb https://repo.download.nvidia.com/jetson/t210 r32.6 main
```
- Reinstall packets
```sh
$ sudo apt update
$ sudo apt install --reinstall nvidia-l4t-gstreamer -y
$ sudo apt install --reinstall nvidia-l4t-multimedia -y
$ sudo apt install --reinstall nvidia-l4t-core -y
$ sudo reboot
```

# Deepstream with Docker
Download the oficial deepstream images from nvidia:\
[Deepstream docker images](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html)

# Install Deepstream
__NOTE:__ For Deepstream 5.0 or 5.1 use the deb package for that version\

- Download the ".deb" package, you will need to login to NVIDIA page:\
https://developer.nvidia.com/deepstream-6.0_6.0.0-1_arm64deb
- Install:
```sh
$ sudo apt-get install -y libgstrtspserver-1.0-0 libgstreamer-plugins-base1.0-dev
$ sudo apt-get install ./deepstream-6.0_6.0.0-1_arm64.deb
```

# Install Deepstream python depedencies (only for deepstream python)
### Instalar numpy y opencv:
```sh
$ sudo apt update
$ sudo apt install python3-numpy python3-opencv -y
```

### Instalar Python Bindings
Reference:\
[Deepstream guide python](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html)

1) Befor install gst-python install:
```sh
$ sudo apt-get install python3-dev libpython3-dev python-gi-dev -y
$ export GST_LIBS="-lgstreamer-1.0 -lgobject-2.0 -lglib-2.0"
$ export GST_CFLAGS="-pthread -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include"
```

2) Download and install git repository of gst-python:
```sh
$ git clone https://github.com/GStreamer/gst-python.git
$ cd gst-python
$ git checkout 1a8f48a
$ git config --global http.sslverify false
$ ./autogen.sh PYTHON=python3
$ ./configure PYTHON=python3
$ make
$ sudo make install
```

3) MetadaClass Access for Python:\
__NOTE:__ For deepstream 5.0 or 5.1, in deepstream will raise an error.
```sh
$cd /opt/nvidia/deepstream/deepstream/lib
$ sudo python3 setup.py install
```

3bis) Install pyds 
__NOTE:__ Run the following steps case that the last point raise an error --> Deepstream 6.0)
- Install dependencies:
```sh
$ sudo apt install -y git python-dev python3 python3-pip python3.6-dev cmake g++ build-essential libglib2.0-dev libglib2.0-dev-bin python-gi-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev pkg-config
```
```sh
pip3 install pycairo
```
- Download repo and install pyds:
```sh
$ git clone --recursive https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
$ cd deepstream_python_apps/bindings/
$ mkdir build
$ cd build
$ cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=6 -DPIP_PLATFORM=linux_aarch64 -DDS_PATH=/opt/nvidia/deepstream/deepstream
$ make
$ pip3 install ./pyds-1.1.0-py3-none*.whl
```

4) Solve issue "libgomp-so-1-cannot-allocate-memory" (deepstream 6.0)
Reference:\
[Foro link reference](https://forums.developer.nvidia.com/t/error-importerror-usr-lib-aarch64-linux-gnu-libgomp-so-1-cannot-allocate-memory-in-static-tls-block-i-looked-through-available-threads-already/166494)

Export library path:
```sh
$ export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

Add to bashrc:
- Open bashrc
```sh
$ sudo nano ~/.bashrc
```
- Add at the end
```sh
$ export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
$ source ~/.bashrc
```

# Download Deepstream python examples
- Download repo:
```sh
$ cd /opt/nvidia/deepstream/deepstream/sources
$ git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
```
- Run example
```sh
$ cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/apps/deepstream-test1
$ python3 deepstream_test_1.py /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264
```
