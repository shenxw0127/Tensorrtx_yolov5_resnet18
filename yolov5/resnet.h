//
// Created by tdt on 2021/5/5.
//

#ifndef COOTRANSFORMSTION_RESNET_H
#define COOTRANSFORMSTION_RESNET_H
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
namespace tdt_resnet {
    class resnet_predictor {
    public:


        resnet_predictor();

        ~resnet_predictor() {
            delete[] trtModelStream;
            context->destroy();
            engine->destroy();
            runtime->destroy();
        }


        int static make_engine();

        int predict(cv::Mat images);

    private:
        char *trtModelStream{nullptr};
        size_t size{0};
        nvinfer1::IRuntime *runtime;
        nvinfer1::ICudaEngine *engine;
        nvinfer1::IExecutionContext *context;

    };
}
#endif //COOTRANSFORMSTION_RESNET_H
