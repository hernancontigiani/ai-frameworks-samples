# Deepstream generic detector example

# Description
In this example testing a generic detector model, translating it to ONNX format and using it in Deepstream.

## Requirements
For the requirements, please refer to the main curated documentation on how to install them.
| Package    | Version |
| ---------- | ------- |
| Jetpack    | 4.5.1   |
| Python     | 3.6.9   |
| Deepstream | 5.1     |
| Tensorflow | 2.4.0   |
| Keras      | 2.4.0   |
| TensorRT   | 7.1.3   |
| tf2onnx    | 1.9.2   |
| ONNX       | 1.10.1  |

## Model setup
In this case the detector model is used as primary infer with nvInfer becuase it is the only model in the pipeline feed by a video, an image or a numpy array.

This examples is using the primary detector models available from deepstream sample models. In order to avoid any relative path issue we suggest you to create the following simbolink link in this folder:
```
sudo ln -s /opt/nvidia/deepstream/deepstream/samples/models ./models
```

## How to run
Feed pipeline with a video (using numpy app-src):
```
python3 deepstream_detector_fakesink.py file://<absolute path>
```

Feed pipeline with a numpy array from a image (use numpy app-src):
```
python3 deepstream_detector_from_numpy.py image.jpg
```

Test FPS of your model from a video (using uribindecode)
```
python3 deepstream_detector_test_fps.py file://<absolute path>
```

Output video detection in out.mp4 (using urbindecode):
```
python3 deepstream_detector_filesink.py file://<absolute path>
```
