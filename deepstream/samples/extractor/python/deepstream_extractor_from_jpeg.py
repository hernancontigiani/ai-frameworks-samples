#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys

import gi
import math

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

import ctypes
import pyds
import os.path
from os import path

import cv2
import numpy as np

MODEL_INPUT_WIDTH = 128
MODEL_INPUT_HEIGHT = 256
MODEL_OUTPUT_VECTOR = 512


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def layer_finder(output_layer_info, name):
    """ 
        Return the layer contained in output_layer_info which corresponds
        to the given name.
        dateType enum:
            FLOAT FP32 format.
            HALF FP16 format.
            INT8 INT8 format.
            INT32 INT32 format.
    """
    for layer in output_layer_info:
        # dataType == 0 <=> dataType == FLOAT
        if layer.dataType == 0 and layer.layerName == name:
            return layer
    return None


def src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        frame_number = frame_meta.frame_num
        l_user = frame_meta.frame_user_meta_list

        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta and user_meta.base_meta.meta_type == \
                    pyds.NVDSINFER_TENSOR_OUTPUT_META:
                try:
                    # Note that user_meta.user_meta_data needs a cast to
                    # pyds.NvDsInferSegmentationMeta
                    # The casting is done by pyds.NvDsInferSegmentationMeta.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone.
                    print("-------------------------------------------------")
                    print("Getting out meta tensor data")
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                    # Get layers info
                    layers_info = []                    
                    print(f"Get output layer info from {tensor_meta.num_output_layers} layers")
                    for i in range(tensor_meta.num_output_layers):
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                        layers_info.append(layer)

                    # API reference
                    # Python
                    # https://docs.nvidia.com/metropolis/deepstream/python-api/index.html
                    output_layer = layer_finder(layers_info, "output")
                    
                    print(f'Get output data from layer "{output_layer.layerName}" of type: {output_layer.dataType}')

                    if not output_layer:
                        print("ERROR: some layers missing in output tensors")
                    
                    # Diferents way of read an output vector:
                    print('1) Read output vector element by element with "get_detections"')
                    vector1 = []
                    for i in range(MODEL_OUTPUT_VECTOR):
                        vector1.append(pyds.get_detections(output_layer.buffer, i))

                    vector1 = np.array(vector1)

                    features_5 = np.load("result.npy")[0]
                    print("Comparacion:", cosine_similarity(vector1, features_5))

                    print("-------------------------------------------------")
                    print("2) Read output vector in one operation")
                    
                    # Cast output vector to float pointer
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    # Create a vector from pointer
                    vector2 = np.ctypeslib.as_array(ptr, shape=(MODEL_OUTPUT_VECTOR,))
                    print("Comparacion:", cosine_similarity(vector1, features_5))
                    print("-------------------------------------------------")

                    #print(vector1)
                    
                except StopIteration:
                    break
               

            try:
                l_user = l_user.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s image_file" % args[0])
        sys.exit(1)

    image_file = args[1]
    num_sources = 1
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # Since the data format in the input file is jpeg,
    # we need a jpegparser
    print("Creating jpegParser \n")
    jpegparser = Gst.ElementFactory.make("jpegparse", "jpeg-parser")
    if not jpegparser:
        sys.stderr.write("Unable to create jpegparser \n")

    # Use nvdec for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Create segmentation for primary inference
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    if not pgie:
        sys.stderr.write("Unable to create primary inferene\n")

    print("Creating fakesink \n")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Playing file", image_file)
    source.set_property('location', image_file)
    if image_file.endswith("mjpeg") or image_file.endswith("mjpg"):
        decoder.set_property('mjpeg', 1)

    # streammux match model width and height to proper scale
    streammux.set_property('width', MODEL_INPUT_WIDTH)
    streammux.set_property('height', MODEL_INPUT_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "dstest_extractor_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != num_sources:
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size,
              " with number of sources ", num_sources,
              " \n")
        pgie.set_property("batch-size", num_sources)


    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(jpegparser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sink)

    # we link the elements together
    # numpy-source -> nvvideoconvert -> capsfilter ->
    # streammux -> nvinfer -> sink
    print("Linking elements in the Pipeline \n")
    source.link(jpegparser)
    jpegparser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the src pad of the inference element
    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, src_pad_buffer_probe, 0)

    # List the sources
    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
