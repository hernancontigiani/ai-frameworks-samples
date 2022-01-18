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
#### Add cuda path to bashsrc:
- Open bashrc:
```sh
sudo nano ~/.bashrc
```
- Add at the end the following lines:
```sh
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```
- Reload configuration:
```sh
source ~/.bashrc
```
- Now check that nvcc get your GPU info:
```sh
nvcc --version
```
#### Install PyCUDA (only for Python)
This could take a couple of minutes:
```sh
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install cython
pip3 install pycuda
```

# Using TRTexec as ONNX-TRT converter
### NOTE ABOUT TensorRT 7.1.2
This version is having problem with dynamic shape, you could see something like this:
```
"InstanceNormalization does not support dynamic inputs"
```
In that case use fix batch size (dynamic_shape=False) when you convert your model to ONNX. You could check that using [netron](https://netron.app/) to visualize your model graph.

### TRTexec parameters
- --fp32 | --fp16 | --int8 → precision (default --fp32)
- --workspace=N → workspace reserved for the model in MB (default 16MB)
- --saveEngine=<some_model.engine> → output engine path

### TRTexec with fixed batch (default=1)
In this case we need a ONNX model without dynamic shape (in this example batch size=1)
```sh
/usr/src/tensorrt/bin/trtexec --onnx=model_b1.onnx --saveEngine=model_b1.trt --workspace=1024
```
### TRTexec with dynamic batch
In this case we need a ONNX model with dynamic shape and defined the profile possible shapes (min, max, opt):
```sh
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.trt --workspace=1024 --minShapes=input:1x3x32x32 --optShapes=input:16x3x32x32 --maxShapes=input:32x3x32x32 --shapes=input:16x3x32x32 --explicitBatch
```

# Test your TRTEngine model with TRTexec
Test your model with a input shape min < shape < max:
```sh
/usr/src/tensorrt/bin/trtexec --loadEngine=model.trt --shapes=input:20x3x32x32
```


# Test your TRTEngine model with Python
Checkout the trt_run.py script example.
