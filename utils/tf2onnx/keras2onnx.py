import onnx
import tf2onnx
from tensorflow.keras.models import load_model

model = load_model("model.h5")

# opese=12 -->  recommended for ONNX runtime, try opset=9 if it fail in TRT
onnx_model, _ = tf2onnx.convert.from_keras(model, inputs_as_nchw="input_1:0", opset=12)
onnx.save(onnx_model, 'model.onnx')