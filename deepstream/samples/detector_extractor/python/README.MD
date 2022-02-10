# Deepstream generic detector & extractor example

# Description
In this example testing a generic detector and extractor model together, translating it to ONNX format and using it in Deepstream.

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

In folder extractor_models you will find two differents generic models:

#### model.onnx
This model gives 512 vector output base on image pixel input:
```
x1 = img[0, :, 0])  # first 128 values of channel R
x2 = img[0, :, 1])  # first 128 values of channel G
x3 = img[0, :, 2])  # first 128 values of channel B
x4 = img[1, :, 0])  # values 128-256 of channel R

out = concatenate(x1, x2, x3, x4)  # vector output size of 512
```
Following consideration has been taken in __config_extractor.txt__:
- num-detected-classes=512  --> 512 is the output vector sieze
- process-mode=2 --> secondary, takes crops output from primary detector.
- operate-on-gie-id=1 --> Take input from gie-id=1 (primary detector).
- network-type=1 --> 1 for classifier, but we are going to use custom user NvDsInferTensorMeta, so no matters this value.
- output-blob-names=output --> tell the system to include the layer "output" in the user NvDsInferTensorMeta
- output-tensor-meta=1 --> enable the output data as user custom tensor meta data

#### flatten.onnx
This model output input image (256x128) as output flatten vector to get the crop image that comes from the detector. The following consideration has been taken in __config_flatten.txt__:
- num-detected-classes=98304  --> flatten output image of size 256x128x3
- other parameters are the same with extractor config example.

## How to run
Feed pipeline with a numpy array from a image (use numpy app-src):
```
python3 deepstream_detector_extractor_from_numpy.py image.jpg
```

Get crop image detected from classes as output file from a original image (use numpy app-src):
```
python3 deepstream_detector_extractor_test_crop.py image.jpg
```

Test FPS of your model from a static image as video (use numpy app-src):
```
python3 deepstream_detector_extractor_test_static.py image.jpg
```
