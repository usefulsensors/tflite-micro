/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the specific model to run
#include "tensorflow/lite/micro/examples/custom_model_rf_keras/custom_model_rf_keras_model_data.h"
#include "tensorflow/lite/micro/examples/custom_model_rf_keras/custom_model_rf_keras_utils.h"
#include "tensorflow/lite/micro/examples/custom_model_rf_keras/settings.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {

  tflite::MicroErrorReporter micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(g_custom_model_rf_keras_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  tflite::MicroMutableOpResolver<11> resolver;
  resolver.AddPad();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddRelu();
  resolver.AddStridedSlice();
  resolver.AddReshape();
  resolver.AddConcatenation();
  resolver.AddAdd();
  resolver.AddSoftmax();
  resolver.AddShape();
  resolver.AddPack();

  constexpr int kTensorArenaSize = 0.3456 * 1024.0 * 1024.0; // MB
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* input = interpreter.input(0);

  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, input->type);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);     // N
  TF_LITE_MICRO_EXPECT_EQ(IMG_H, input->dims->data[1]); // H
  TF_LITE_MICRO_EXPECT_EQ(IMG_W, input->dims->data[2]); // W
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[3]);     // C

  //std::string src_path = __FILE__;
  //std::string src_dir = src_path.substr(0, src_path.rfind("/"));
  //std::string img_path = cv::samples::findFile(
  //    src_dir + "/images/test-img.jpeg");
  //cv::Mat raw_img = cv::imread(img_path, cv::IMREAD_COLOR);
  //cv::Mat img;
  //cv::resize(raw_img, img, cv::Size(IMG_W, IMG_H), 0.0, 0.0, cv::INTER_LINEAR);

  // img is in HWC format -> need to convert to NHWC
  // also unroll image
  for (int vH = 0; vH < IMG_H; vH++) {
    for (int vW = 0; vW < IMG_W; vW++) {
      //int vv = img.at<uchar>(vH*IMG_W + vW);
      uint8_t v = 17;
      input->data.f[vW + vH*IMG_W] = v;
    }
  }

  TfLiteStatus invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  std::cout << "model executed!" << std::endl;

  TfLiteTensor* output0 = interpreter.output(0);
  TfLiteTensor* output1 = interpreter.output(1);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output0->type);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output1->type);
  TF_LITE_MICRO_EXPECT_EQ(3, output0->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(3, output1->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output0->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output1->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(2, output0->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(4 , output1->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(TOTAL_ANCHORS, output0->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(TOTAL_ANCHORS, output1->dims->data[1]);

}

TF_LITE_MICRO_TESTS_END
