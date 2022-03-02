/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>

#include <math.h>

#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"
#include <iostream>

#include "ieee_half.h"
typedef half_float::half float16;

#include <opencv2/objdetect/objdetect.hpp>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"
#include <chrono>

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"

#define INFER_PGIE_CONFIG_FILE  "config_infer_primary_facenet.txt"
#define INFER_SGIE_CONFIG_FILE "config_sgie_age_gender.txt"

#define MAX_DISPLAY_LEN 64

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 800
#define MUXER_OUTPUT_HEIGHT 514


/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;

/* nvds_lib_major_version and nvds_lib_minor_version is the version number of
 * deepstream sdk */

unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;


class ClockChrom {
public:
    ClockChrom() { start(); };

    inline void start()
    {
	using namespace std::chrono;
        t1 = high_resolution_clock::now();
    }

    inline void reset()
    {
        start();
    }

    inline void stop()
    {
	using namespace std::chrono;
        t2 = high_resolution_clock::now();
    }

    template <typename T, typename... Args>
    inline void execute(T f, Args... args)
    {
        start();
        f(args...);
        stop();
    }

    template <typename T>
    inline double getDuration()
    {
        stop();
        return std::chrono::duration_cast<T>(t2 - t1).count();
    }

private:
    std::chrono::high_resolution_clock::time_point t1, t2;
};

extern "C"
    bool NvDsInferParseCustomResnet (std::vector < NvDsInferLayerInfo >
    const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector < NvDsInferObjectDetectionInfo > &objectList);

/* This is the buffer probe function that we have registered on the sink pad
 * of the tiler element. All SGIE infer elements in the pipeline shall attach
 * their NvDsInferTensorMeta to each object's metadata of each frame, here we will
 * iterate & parse the tensor data to get classification confidence and labels.
 * The result would be attached as classifier_meta into its object's metadata.
 */
static GstPadProbeReturn
sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  static guint use_device_mem = 0;
  static size_t frame_number = 0;
  static ClockChrom chrm;
  static std::size_t countFrames = 0;
  static float fps = 0;

  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  std::cout << "--------------------------------\n";
  std::cout << "Frame number: " << frame_number << '\n';
  frame_number++;

  auto duration = chrm.getDuration<std::chrono::milliseconds>();
  decltype(duration) second = 1000;

  if (duration > second){
    decltype(fps) tmpFPS = countFrames / (duration * 1E-3);
    fps = fps > 0 ? (fps + tmpFPS) / 2 : tmpFPS;
    std::cout << "--------------------------------\n";
    std::cout << "--------------------------------\n";
    std::cout << "FPS: " << fps << '\n';
    std::cout << "--------------------------------\n";
    std::cout << "--------------------------------\n";

    countFrames = 0;
    chrm.reset();
  }

  countFrames++;

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      std::cout << "\t";
      std::cout << "Object id: " << static_cast<int>(obj_meta->object_id) << "\n";

      float confidence = static_cast<float>(obj_meta->confidence);
      int class_id = static_cast<int>(obj_meta->class_id);
      int top = static_cast<int>(obj_meta->rect_params.top);
      int left = static_cast<int>(obj_meta->rect_params.left);
      int width = static_cast<int>(obj_meta->rect_params.width);
      int height = static_cast<int>(obj_meta->rect_params.height);

      std::cout << "type " << class_id << " score "<< confidence << " box " << left << " " << top << " " << width << " " << height << "\n";

      // Iterate user metadata in object to search SGIE's tensor data
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
          l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
          continue;

        // convert to tensor metadata
        NvDsInferTensorMeta *meta =
            (NvDsInferTensorMeta *) user_meta->user_meta_data;

        for (unsigned int i = 0; i < meta->num_output_layers; i++) {
          NvDsInferLayerInfo *info = &meta->output_layers_info[i];
          info->buffer = meta->out_buf_ptrs_host[i];

          std::cout << "\t\t";
	      std::cout << "Layer name: " << info->layerName << " numElements: " << info->inferDims.numElements <<  '\n';

          if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
            cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                info->inferDims.numElements * 2, cudaMemcpyDeviceToHost);
          }    

         auto* ptr = reinterpret_cast<float16*>(info->buffer);

         for (size_t j = 0; j < info->inferDims.numElements; j++) {
	          std::cout << "\t\t\t";
	          std::cout << "Tensor " << j << ": " << ptr[j] << std::endl;
         }
                    
        }
      }
    }
  }

  use_device_mem = 1 - use_device_mem;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
usage(const char *bin)
{
  g_printerr
    ("Usage: %s [-t infer-type]<elementary H264 file 1> ... <elementary H264 file n>\n",
      bin);
  g_printerr
    ("     -t infer-type: select form [infer, inferserver], infer by default\n");
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =
      NULL, *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie =
      NULL, *nvvidconv = NULL, *nvosd = NULL, *sgie;
  g_print ("With tracker\n");
  GstElement *transform = NULL;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL, *pgie_src_pad = NULL, *tiler_sink_pad = NULL;
  guint i = 0;
  std::vector<std::string> files;
  gboolean is_nvinfer_server = FALSE;
  const char *infer_plugin = NVINFER_PLUGIN;

  /* Parse infer type and file names */
  for (gint k = 1; k < argc;) {
    if (!strcmp("-t", argv[k])) {
      if (k + 1 >= argc) {
        usage(argv[0]);
        return -1;
      }
      if (!strcmp("infer", argv[k+1])) {
        is_nvinfer_server = false;
      } else if (!strcmp("inferserver", argv[k+1])) {
        is_nvinfer_server = true;
      } else {
        usage(argv[0]);
        return -1;
      }
      k += 2;
    } else {
      files.emplace_back (argv[k]);
      ++k;
    }
  }
  /* Check input files */
  if (files.empty()) {
    usage(argv[0]);
    return -1;
  }
  guint num_sources = files.size();

  nvds_version(&nvds_lib_major_version, &nvds_lib_minor_version);

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */

  /* Create Pipeline element that will be a container of other elements */
  pipeline = gst_pipeline_new ("dstensor-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make (infer_plugin, "primary-nvinference-engine");

  GstElement *x264enc = NULL, *qtmux = NULL;

  queue = gst_element_factory_make ("queue", NULL);
  
  /* We need three secondary gies so lets create 3 more instances of
     nvinfer */
  sgie = gst_element_factory_make (infer_plugin, "secondary-nvinference-engine");

  transform = gst_element_factory_make("identity", "identity");
  sink = gst_element_factory_make ("fakesink", "sink");

  if (!pgie || !sgie || !sink || !transform ) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set all the necessary properties of the infer plugin element,
   * Enable Output tensor meta, we can probe PGIE and
   * SGIEs buffer to parse tensor output data of models */
    /* nvinfer Output tensor meta can be enabled by set "output-tensor-meta=true"
     * here or enable this attribute in config file. */
  g_object_set (G_OBJECT (pgie), "config-file-path", INFER_PGIE_CONFIG_FILE,
      "output-tensor-meta", FALSE, "batch-size", num_sources, NULL);
  g_object_set (G_OBJECT (sgie), "config-file-path", INFER_SGIE_CONFIG_FILE, NULL);
  /*g_object_set (G_OBJECT (sgie), "config-file-path", INFER_SGIE_CONFIG_FILE,
      "output-tensor-meta", TRUE, "process-mode", 2, NULL);*/



  g_object_set(G_OBJECT(sink), "sync", 0, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  /* decoder | pgie1 | sgie1 | sgie2 | sgie3 | etc.. */
  gst_bin_add_many (GST_BIN (pipeline),
      streammux, pgie, queue, sgie, sink, NULL);

  for (i = 0; i < num_sources; i++) {
    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", NULL);

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    h264parser = gst_element_factory_make ("h264parse", NULL);

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", NULL);
    gst_bin_add_many (GST_BIN (pipeline), source, h264parser, decoder, NULL);

    if (!source || !h264parser || !decoder) {
      g_printerr ("One element could not be created. Exiting.\n");
      return -1;
    }

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16];
    sprintf (pad_name_sink, "sink_%d", i);
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
      g_printerr ("Decoder request src pad failed. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    /* Link the elements together */
    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
      g_printerr ("Elements could not be linked: 1. Exiting.\n");
      return -1;
    }
    /* Set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", files[i].c_str(), NULL);
  }

  if (!gst_element_link_many (streammux, pgie, queue, sgie, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Add probe to get informed of the meta data generated, we add probe to
   * the source pad of PGIE's next queue element, since by that time, PGIE's
   * buffer would have had got tensor metadata. */
  /*pgie_src_pad = gst_element_get_static_pad (queue, "src");
  gst_pad_add_probe (pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      sgie_pad_buffer_probe, NULL, NULL);*/

    GstPad *sinkpad = gst_element_get_static_pad (sink, "sink");
  gst_pad_add_probe (sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
      sgie_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing...\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Iterate */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
