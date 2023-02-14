//
// Created by tdt on 2021/1/24.
//

#ifndef YOLOV5_YOLOV5_PREDICTOR_H
#define YOLOV5_YOLOV5_PREDICTOR_H
#include <opencv2/opencv.hpp>
#include "NvInferRuntime.h"
#include "resnet.h"
namespace tdt_yolov5_predictor {
    class yolov5_predictor{
    public:
        yolov5_predictor();
        tdt_resnet::resnet_predictor *resnet;
        static void make_engine();

//        ~yolov5_predictor();
    private:

        float *prob;
        float* buffers[2];
        nvinfer1::IExecutionContext *context;
        cudaStream_t stream;
        uint8_t* img_host;
        uint8_t* img_device;
        nvinfer1::IRuntime *runtime;
        nvinfer1::ICudaEngine *engine;
        int inputIndex;
        int outputIndex;

    };
    inline static bool RectSafety(cv::Rect2f &rect,const cv::Size2f &size) {
        // cv::Rect2f out_rect=cv::Rect2f(0,0,size.width,size.height);
        // out_rect=rect&out_rect;
        // return !out_rect.area()<rect.area();
        if(rect.x>0&&rect.y>0&&(rect.x+rect.width)<size.width&&(rect.y+rect.height)<size.height){
            return true;
        }else{
            return false;
        }
    }

}


#endif //YOLOV5_YOLOV5_PREDICTOR_H
