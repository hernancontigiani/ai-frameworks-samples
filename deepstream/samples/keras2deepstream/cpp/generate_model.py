import time
t1 = time.time()

import keras2onnx
import onnx
import tf2onnx.convert
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet

model = MobileNet(include_top=True, weights=None, input_shape=(3,224,224), classes=3)

onnx_model, _ = tf2onnx.convert.from_keras(model, inputs_as_nchw="input_1:0")

onnx.save(onnx_model, 'models/model.onnx')

t2 = time.time()

print(f"Finished converting model... it took {t2-t1} seconds")
