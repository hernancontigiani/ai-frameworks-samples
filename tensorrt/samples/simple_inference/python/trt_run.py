import os
import time

import tensorrt as trt
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit


# Implcit vs Explicit batch
# https://forums.developer.nvidia.com/t/whats-the-differnece-between-implicit-and-explicit-batch-in-tensor-rt/144946
# todo lo que venga de onnx será con explicit batch
# si algo viene de etlt puede que venga con implicit batch

# Inference with implicit vs explicit batch
# https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/ExecutionContext.html?highlight=execute_async#tensorrt.IExecutionContext.execute_async_v2

# Multi stream / contexto for paralell inference
# https://github.com/NVIDIA/TensorRT/issues/1430


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:    
    def __init__(self, engine_path, max_batch_size=None):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        
        if self.engine.has_implicit_batch_dimension == False:
            # Batch explicit, that it from profile shape
            # From the bind 0 (input) first profile (0) --> get_profile_shape
            # Returns de max batch size of all profiles shapes --> [:, 0]
            self.max_batch_size = int(max(max(np.array(self.engine.get_profile_shape(0, 0))[:, 0]), 1))
        else:
            # Batch implicit, that it from engine max batch definition
            self.max_batch_size = self.engine.max_batch_size
        
        if max_batch_size is not None and self.max_batch_size > max_batch_size:
            self.max_batch_size = max_batch_size
        
        self.inputs, self.inputs_shape, self.inputs_dtype, self.outputs, self.outputs_shape, self.outputs_dtype, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
        # BUG: avoid double free error at context delete (TrtModel object delete)
        # So the Python garbage collector will leave it alone
        self.__out = []

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        ''' Deserialization of cuda engine'''
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        ''' Reserve cuda (GPU) memory space for input and output (bindings)'''
        inputs = []
        inputs_shape = []
        inputs_dtype = []
        outputs = []
        outputs_shape = []
        outputs_dtype = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            if self.engine.has_implicit_batch_dimension == False:
                # Explicit batch, first shape value is the batch dimension
                shape = self.engine.get_binding_shape(binding)[1:]
            else:
                shape = self.engine.get_binding_shape(binding)

            #shape = trt.volume(np.squeeze(shape))
            size = trt.volume(shape) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
                inputs_shape.append(shape)
                inputs_dtype.append(dtype)
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                outputs_shape.append(shape)
                outputs_dtype.append(dtype)
        
        return inputs, inputs_shape, inputs_dtype, outputs, outputs_shape, outputs_dtype, bindings, stream
    
    def __do_inference(self, batch_size=1):
        ''' GPU Inference'''
        # Copy data to GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        # Make inference
        # Asynchronously execute inference on a batch
        if self.engine.has_implicit_batch_dimension == False:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            # execute_async_v2 not support implicit batch
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)        

        # Wait for finish and get data from GPU memory
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        # Get output data
        self.stream.synchronize()
        self.__out =  [self.outputs[i].host.reshape(self.max_batch_size, *self.outputs_shape[i]) for i in range(len(self.outputs))]

    def __call__(self, x:np.ndarray, batch_size=1):
        ''' GPU Batch Inference

            TODO: Only work with model with one input
        '''
        # Copy data to GPU input buffer memory
        # TODO: Only work with model with one input
        x = x.astype(self.inputs_dtype[0])

        y = []
        for i in range(len(self.outputs_shape)):
            y.append(np.zeros((batch_size, *self.outputs_shape[i])).astype(self.outputs_dtype[i]))

        # Define the min batch operation
        min_batch = min(batch_size, self.max_batch_size)

        # Set the batch size of each input
        for i in range(len(self.inputs_shape)):
            if self.engine.has_implicit_batch_dimension == False:
                self.context.set_binding_shape(i, (min_batch, *self.inputs_shape[i]))
            else:
                # implicit batch --> batch size is declared during inference
                self.context.set_binding_shape(i, self.inputs_shape[i])

        last_batch = 0
        batch = 0
        # Do X inference un step of min_batch until batch_size
        for batch in range(min_batch, batch_size, min_batch):
            # Get X data to process in min_batch size
            self.inputs[0].host = x.data[last_batch: batch]        
            self.__do_inference(min_batch)

            # Get each output and add to the "y" result vector
            for i in range(len(self.__out)):
                y[i][last_batch: batch] = self.__out[i][:(batch-last_batch)]
            last_batch = batch

        # In case that are pending some inference, make them now
        # This ocurrs when --> max_batch_size > batch_size
        # Or batch_size and max_batch_size are not multiples
        if batch < batch_size:
            last_batch = batch
            batch = batch_size

            # Get X data to process in min_batch size
            # TODO: Only work with model with one input
            self.inputs[0].host = x.data[last_batch: batch]        
            self.__do_inference(batch-last_batch)

            # Get each output and add to the "y" result vector
            for i in range(len(self.__out)):
                y[i][last_batch: batch] = self.__out[i][:(batch-last_batch)]
            last_batch = batch

        return y

       
if __name__ == "__main__":
    batch_size = 16  # number in inputs
    trt_engine_path = "model.trt"  # your TRTEngine model
    model = TrtModel(trt_engine_path)

    print("Model input shape:", model.inputs_shape)
    print("Model output shape:", model.outputs_shape)
    print("Model max batch size:", model.max_batch_size)

    input_data = np.random.randint(0, 255, (batch_size, *model.inputs_shape[0])) / 255
    print("Input data shape:", input_data.shape)

    print("Test performance:")
    for i in range(10):
        t1 = time.time()
        result = model(input_data, batch_size)[0] # get first output
        t2 = time.time()
        print(f"Time: {(t2-t1)*1000:.2f}ms, {1/(t2-t1):.2f}FPS")