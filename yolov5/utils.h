#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/opencv.hpp>

cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = Yolo::INPUT_W / (img.cols*1.0);
    float r_h = Yolo::INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = Yolo::INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (Yolo::INPUT_H - h) / 2;
    } else {
        w = r_h * img.cols;
        h = Yolo::INPUT_H;
        x = (Yolo::INPUT_W - w) / 2;
        y = 0;
    }


    cv::resize(img, img, cv::Size(w,h), 0, 0, cv::INTER_LINEAR);auto start = std::chrono::system_clock::now();
    cv::Mat out(Yolo::INPUT_H, Yolo::INPUT_W, CV_8UC3);
    auto end = std::chrono::system_clock::now();
    img.copyTo(out(cv::Rect(x, y, img.cols, img.rows)));


    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms 图片转换时间"<<std::endl;
    return out;
}

#endif  // TRTX_YOLOV5_UTILS_H_

