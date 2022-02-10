# TensorRT simple inference
This script show how to run a generic TRT inference, base on the model.trt build with the util "onnx2tensorrt"

## How to run
- You will need a trtEngine model, in case you dont have you, you could create it using the utils "pytorch2onnx" and "onnx2tensort" examples.
```sh
$ python3 trt_run.py
```

## References
#### Implcit vs Explicit batch
https://forums.developer.nvidia.com/t/whats-the-differnece-between-implicit-and-explicit-batch-in-tensor-rt/144946
- Todo lo que venga de onnx será con explicit batch
- Si algo viene de etlt puede que venga con implicit batch

##### Inference with implicit vs explicit batch
https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html?highlight=execute_async#tensorrt.IExecutionContext.execute_async_v2

#### Multi stream / contexto for paralell inference
https://github.com/NVIDIA/TensorRT/issues/1430