# Deepstream pose estimation
Reference:
https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation

# Download the repo and compile:
#### Clone repo in /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps to avoid dependencie issues:
```sh
$ git clone https://github.com/NVIDIA-AI-IOT/deepstream_pose_estimation
$ cd deepstream-pose-estimation/
```

#### Edit "Makefile" to change deepstream version:
```
NVDS_VERSION:=5.1
```

#### Compile:
```sh
$ sudo make
```

NOTE: In case that system fail because it didnt found gst.h, try install this package:
```sh
$ sudo apt install libjson-glib-dev
```

# Model to TRT
In the repo, it recommends to download the weights ".pth" model from the TRTPose repo and convert to ONNX using the script available, but this this model the system file. Inted used the pose_estimation.onnx available on the repo.

Conver to TRT:
```sh
$ /usr/src/tensorrt/bin/trtexec --onnx=pose_estimation.onnx --saveEngine=pose_estimation.onnx_b1_gpu0_fp16.engine
```

# Test
NOTE: The repo only works with H264 videos, there is a pull-request with the fix to this issue.

Excecute:
```sh
$ sudo ./deepstream-pose-estimation-app <file-uri> <output-path>
```

Example:
```sh
$ sudo ./deepstream-pose-estimation-app mi_video.mp4 ./output/
```


