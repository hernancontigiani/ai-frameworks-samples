# Keras to Deepstream with Mobilenet as a classifier

# Description
In this example we are creating an untrained classifier Mobilenet model with Keras, translating it to ONNX format and using it in Deepstream.

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

## How to run
After getting all the dependencies installed, we are going to execute the following steps:
1. Change channel setting in ~/.keras/keras.json
    {
        "floatx": "float32",
        "epsilon": 1e-07,
        "backend": "tensorflow",
        "image_data_format": "channels_first"
    }
2. Generate new model
    ```sh
    python3 generate_model.py
    ```
3. Compile binary
   ```sh
      make
   ```

   note: you might want to use sudo since it's going to read some protected files in deepstream directory.

4. Execute binary with video example
    ```sh
      ./deepstream-infer-tensor-meta-app /opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_720p.h264
    ```

# Considerations
