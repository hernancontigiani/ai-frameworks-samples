import os
import time

import tensorrt as trt
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:    
    def __init__(self, engine_path, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)

        # From the bind 0 (input) first profile (0) --> get_profile_shape        
        # Returns de max batch size of all profiles shapes --> [:, 0]        
        self.max_batch_size = int(max(max(np.array(self.engine.get_profile_shape(0, 0))[:, 0]), 1))
        print("Model max batch size:", self.max_batch_size)

        self.inputs, self.inputs_shape, self.outputs, self.outputs_shape, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

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
        outputs = []
        outputs_shape = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)[1:]
            size = trt.volume(shape) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
                inputs_shape.append(shape)
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                outputs_shape.append(shape)
        
        return inputs, inputs_shape, outputs, outputs_shape, bindings, stream
    
    def __do_inference(self):
        ''' GPU Inference'''
        # Copy data to GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        # Make inference
        # Asynchronously execute inference on a batch
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Wait for finish and get data from GPU memory
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        # Get output data
        self.stream.synchronize()

        return [out.host.reshape(self.max_batch_size, -1) for out in self.outputs]

            
    def __call__(self, x:np.ndarray, batch_size=1):
        ''' GPU Batch Inference'''
        # Copy data to GPU input buffer memory
        x = x.astype(self.dtype)

        y = []
        for i in range(len(self.outputs_shape)):
            y.append(np.zeros((batch_size, *self.outputs_shape[i])).astype(np.float32))

        # Define the min batch operation
        min_batch = min(batch_size, self.max_batch_size)

        # Set the batch size of each input
        for i in range(len(self.inputs_shape)):
            self.context.set_binding_shape(0, (min_batch, *self.inputs_shape[i]))

        last_batch = 0
        batch = 0
        # Do X inference un step of min_batch until batch_size
        for batch in range(min_batch, batch_size, min_batch):
            # Get X data to process in min_batch size
            self.inputs[0].host = x.data[last_batch: batch]        
            out = self.__do_inference()

            # Get each output and add to the "y" result vector
            for i in range(len(out)):
                y[i][last_batch: batch] = out[i][:(batch-last_batch)]
            last_batch = batch

        # In case that are pending some inference, make them now
        # This ocurrs when --> max_batch_size > batch_size
        # Or batch_size and max_batch_size are not multiples
        if batch < batch_size:
            last_batch = batch
            batch = batch_size

            # Get X data to process in min_batch size
            self.inputs[0].host = x.data[last_batch: batch]        
            out = self.__do_inference()

            # Get each output and add to the "y" result vector
            for i in range(len(out)):
                y[i][last_batch: batch] = out[i][:(batch-last_batch)]
            last_batch = batch

        return y

       
if __name__ == "__main__":
    batch_size = 16  # number in inputs
    trt_engine_path = "model.trt"  # your TRTEngine model
    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)

    input_data = np.random.randint(0, 255, (batch_size, *shape[1:])) / 255
    print("Input data shape:", input_data.shape)

    print("Test performance:")
    for i in range(10):
        t1 = time.time()
        result = model(input_data, batch_size)[0] # get first output
        t2 = time.time()
        print(f"Time: {(t2-t1)*1000:.2f}ms, {1/(t2-t1):.2f}FPS")