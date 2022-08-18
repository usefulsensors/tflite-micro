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

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include the specific model to run
#include "tensorflow/lite/micro/examples/id_model/id_model_model_data.h"
#include "tensorflow/lite/micro/examples/id_model/settings.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {

  tflite::MicroErrorReporter micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(g_id_model_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  tflite::MicroMutableOpResolver<15> resolver;
  resolver.AddPad();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddPrelu();
  resolver.AddConcatenation();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddReshape();
  resolver.AddTranspose();
  resolver.AddSplit();
  resolver.AddExpandDims();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddDequantize();

  constexpr int kTensorArenaSize = 0.5 * 1024.0 * 1024.0; // MB
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

  // img is in HWC format -> need to convert to NHWC; also unroll image
  for (int vH = 0; vH < IMG_H; vH++) {
    for (int vW = 0; vW < IMG_W; vW++) {
      input->data.int8[vW + vH*IMG_W] = 42;
    }
  }

  TfLiteStatus invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  std::cout << "model executed!" << std::endl;

  TfLiteTensor* output0 = interpreter.output(0);

  //TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt8, output0->type);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output0->type);
  TF_LITE_MICRO_EXPECT_EQ(3, output0->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output0->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output0->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(EMB_SZ, output0->dims->data[2]);

}

TF_LITE_MICRO_TESTS_END
