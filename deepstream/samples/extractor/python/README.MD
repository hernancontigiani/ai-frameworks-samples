# Deepstream example with generic extractor model
![](https://developer.nvidia.com/sites/default/files/deepstreamsdk3-workflow.png)
# Description
In this example testing a generic features extractor (embedding extractor) model, translating it to ONNX format and using it in Deepstream.\
In this case the model is takes pictures of 256x128 (heightxwidth) and extract from there 512 values (embedding 512 size) in this way:\
```
x1 = img[0, :, 0])  # first 128 values of channel R
x2 = img[0, :, 1])  # first 128 values of channel G
x3 = img[0, :, 2])  # first 128 values of channel B
x4 = img[1, :, 0])  # values 128-256 of channel R

out = concatenate(x1, x2, x3, x4)  # vector output size of 512
```

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
In this case the extractor model is used as primary infer with nvInfer becuase it is the only model in the pipeline feed by an image or a numpy array. The following consideration have been taken:
- num-detected-classes=512  --> 512 is the output vector sieze
- process-mode=1 --> primary
- network-type=1 --> 1 for classifier, but we are going to use custom user NvDsInferTensorMeta, so no matters this value
- output-blob-names=output --> tell the system to include the layer "output" in the user NvDsInferTensorMeta
- output-tensor-meta=1 --> enable the output data as user custom tensor meta data

## Model creation
This model was created with the notebook bypass.ipynb

## How to run
In the model folder you will find the generic extract model converted, but you could replace with yours.

Feed pipeline with a jgp image (use jpeg decoder and file src):
```
python3 deepstream_extractor_from_jpeg.py image.jpg
```

Feed pipeline with a numpy array from a image (use numpy app-src):
```
python3 deepstream_extractor_from_numpy.py image.jpg
```

Test FPS of your model from a numpy array
```
python3 deepstream_extractor_test_fps.py image.jpg
```
