# Before build any image
We recommend to set de nvidia runtime as defualt. Use the script "set_nvidia_runtime.sh" available inside scripts folder.

# TRTpose docker image
Reference:\
https://github.com/NVIDIA-AI-IOT/jetson-pose-container/blob/main/Dockerfile

Build image:
```sh
$ sudo docker build -t l4t_ml_trtpose -f l4t_ml_trtpose .
```

Run jupyter:
```sh
sudo docker run -it --rm -p 8888:8888 -v $(pwd)/models:/pose_workdir/trt_pose/tasks/human_pose/models l4t_ml_trtpose jupyter lab --NotebookApp.token='' --NotebookApp.password='' --allow-root --ip 0.0.0.0
```
