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

def extractor_src_pad_buffer_probe(pad, info, u_data):
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
        num_obj_meta = frame_meta.num_obj_meta
        print("---------------------------------------------")
        print("Extractor: number of objects", num_obj_meta)
      
        l_obj=frame_meta.obj_meta_list

        while l_obj:
            try: 
                # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                # The casting is done by pyds.NvDsObjectMeta.cast()
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            confidence = float(obj_meta.confidence)
            rect_params = obj_meta.rect_params
            top = int(rect_params.top)
            left = int(rect_params.left)
            width = int(rect_params.width)
            height = int(rect_params.height)

            print(f"type {obj_meta.class_id} score {confidence:.2f} box ({left},{top},{width},{height})")

            l_user = obj_meta.obj_user_meta_list
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
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                        # Get layers info
                        layers_info = []                    
                        for i in range(tensor_meta.num_output_layers):
                            layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                            layers_info.append(layer)

                        # API reference
                        # Python
                        # https://docs.nvidia.com/metropolis/deepstream/python-api/index.html
                        output_layer = layer_finder(layers_info, "output")
                        
                        if not output_layer:
                            print("ERROR: some layers missing in output tensors")

                        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                        # Create a vector from pointer
                        vector = np.ctypeslib.as_array(ptr, shape=(MODEL_OUTPUT_VECTOR,))
                        print("output vector of size", vector.shape)
                        
                    except StopIteration:
                        break
                

                try:
                    l_user = l_user.next
                except StopIteration:
                    break
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK



def detector_src_pad_buffer_probe(pad, info, u_data):
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
        l_obj = frame_meta.obj_meta_list
        num_obj_meta = frame_meta.num_obj_meta
        print("---------------------------------------------")
        print("Detector: number of objects", num_obj_meta)

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            confidence = float(obj_meta.confidence)
            rect_params = obj_meta.rect_params
            top = int(rect_params.top)
            left = int(rect_params.left)
            width = int(rect_params.width)
            height = int(rect_params.height)
            print(f"type {obj_meta.class_id} score {confidence:.2f} box ({left},{top},{width},{height})")

            try:
                l_obj = l_obj.next
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

    # Source element for reading input raw data as frame
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("appsrc", "numpy-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nv-videoconv")        
    if not nvvideoconvert:
        sys.stderr.write(" error nvvid1")

    caps_filter = Gst.ElementFactory.make("capsfilter", "capsfilter1")
    if not caps_filter:
        sys.stderr.write(" error capsf1")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Create segmentation for primary inference
    detector = Gst.ElementFactory.make("nvinfer", "detector-nvinference")
    if not detector:
        sys.stderr.write("Unable to create primary detector\n")
    detector.set_property('config-file-path', "detector_config.txt")

    extractor = Gst.ElementFactory.make("nvinfer", "extractor-nvinference")
    if not extractor:
        sys.stderr.write("Unable to create primary extractor\n")
    extractor.set_property('config-file-path', "config_extractor.txt")

    print("Creating fakesink \n")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    print("Playing file", image_file)

    # Push buffer and check
    img = cv2.imread(image_file)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img[:, :, 3] = 1

    print("input shape:", img.shape)

    input_height, input_width, _ = img.shape

    caps_in = Gst.Caps.from_string(f"video/x-raw,format=RGBA,width={input_width},height={input_height},framerate=20/1")
    caps = Gst.Caps.from_string(f"video/x-raw(memory:NVMM),format=NV12,width={input_width},height={input_height},framerate=20/1")
    source.set_property('caps', caps_in)
    caps_filter.set_property('caps', caps)

    # streammux match model width and height to proper scale
    streammux.set_property('width', input_width)
    streammux.set_property('height', input_height)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    
    pgie_batch_size = detector.get_property("batch-size")
    if pgie_batch_size != num_sources:
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size,
              " with number of sources ", num_sources,
              " \n")
        detector.set_property("batch-size", num_sources)


    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(nvvideoconvert)
    pipeline.add(caps_filter)
    pipeline.add(streammux)
    pipeline.add(detector)
    pipeline.add(extractor)
    pipeline.add(sink)

    # we link the elements together
    # numpy-source -> nvvideoconvert -> capsfilter ->
    # streammux -> nvinfer -> sink
    print("Linking elements in the Pipeline \n")
    source.link(nvvideoconvert)
    nvvideoconvert.link(caps_filter)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_filter.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(detector)
    detector.link(extractor)
    extractor.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the src pad of the inference element
    detector_src_pad = detector.get_static_pad("src")
    if not detector_src_pad:
        sys.stderr.write(" Unable to get detector_src_pad\n")
    else:
        detector_src_pad.add_probe(Gst.PadProbeType.BUFFER, detector_src_pad_buffer_probe, 0)

    extractor_src_pad = extractor.get_static_pad("src")
    if not extractor_src_pad:
        sys.stderr.write(" Unable to get extractor_src_pad\n")
    else:
        extractor_src_pad.add_probe(Gst.PadProbeType.BUFFER, extractor_src_pad_buffer_probe, 0)

    # List the sources
    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)

    def ndarray_to_gst_buffer(array: np.ndarray) -> Gst.Buffer:
        """Converts numpy array to Gst.Buffer"""
        return Gst.Buffer.new_wrapped(array.tobytes())

    source.emit("push-buffer", ndarray_to_gst_buffer(img))
    source.emit("end-of-stream")

    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
