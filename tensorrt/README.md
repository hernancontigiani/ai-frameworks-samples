# TensorRT scripts
![](https://d15shllkswkct0.cloudfront.net/wp-content/blogs.dir/1/files/2019/09/NV_TensorRT_Visual_2C_RGB-625x625-1.png)

# Description
In this folder you will find how to convert your ONNX model to TRTEngine and how to test it without DeepStream:
| Package    | Version   |
| ---------- | -------   |
| Python     | >=3.6.9   |
| Jetpack    | >=4.5.1   |
| TensorRT   | >=7.1.3   |
| pycuda     | >=2021.1  |

All of this is already installed on your Jetpack.

# Requirements
In order to use TensorRT with Python we need first to define the PATH cuda:
### Add cuda path to bashsrc:
- Open bashrc:
```sh
sudo nano ~/.bashrc
```
- Add at the end the following lines:
```sh
$ export PATH=${PATH}:/usr/local/cuda/bin
$ export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```
- Reload configuration:
```sh
$ source ~/.bashrc
```
- Now check that nvcc get your GPU info:
```sh
$ nvcc --version
```
#### Install PyCUDA (only for Python)
This could take a couple of minutes:
```sh
$ sudo apt-get update
$ sudo apt-get install -y python3-pip
$ pip3 install cython
$ pip3 install pycuda
```
