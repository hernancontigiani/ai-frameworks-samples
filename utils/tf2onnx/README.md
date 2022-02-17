# ONNX converter
![](https://keras.io/img/logo.png)

# Description
In this folder you will find scripts to converts from keras and tensorflow models to ONNX. This scripts and platform independent, so the could be exec in your local computer, colab or edge device without any issue of compatibility. The minor versions needed are:
| Package    | Version   |
| ---------- | -------   |
| Python     | >=3.6.9   |
| Tensorflow | >=2.4.0   |
| Keras      | >=2.4.0   |
| tf2onnx    | >=1.9.2   |
| ONNX       | >=1.4.1   |

### Install tf2onnx and onnx:
```sh
$ python3 -m pip install tf2onnx
```

# Scripts
## keras2onnx.py
Use this as example to convert your keras model to ONNX. We recommend to use "opset=12" to use this model on ONNX Runtime.

## age_gender_to_onnx.ipynb
Python notebook with an example of a real Keras model (age and gender prediction) exported to ONNX