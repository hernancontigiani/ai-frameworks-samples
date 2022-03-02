# Deepstream Facenet + AgeGender recognition
![](https://developer.nvidia.com/sites/default/files/deepstreamsdk3-workflow.png)
# Description
In this example testing a generic detector and extractor model together, translating it to ONNX format and using it in Deepstream.

# Requirements
For the requirements, please refer to the main curated documentation on how to install them.
| Package    | Version |
| ---------- | ------- |
| Jetpack    | 4.6     |
| Deepstream | 6.0     |
| TensorRT   | 0.x     |

# Model setup
### Age gender model
Download from:
```

```
### Facenet model
Download facenet model (.etlt) from NVIDIA:
```sh
$ cd ./models
$ wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0/files/model.etlt -O facenet.etlt
```

This links were taken from deepstream tao examples:

* [Deepstream Tao download models](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/download_models.sh)


## How to run
Build the application:
```sh
make
```

Feed pipeline with a video:
```sh
$ ./deepstream_app_from_video people.mp4
```


