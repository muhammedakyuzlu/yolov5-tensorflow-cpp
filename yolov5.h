#pragma once

#include "tensorflow/cc/saved_model/loader.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>



struct Prediction
{
	std::vector<cv::Rect> boxes;
	std::vector<float>    scores;
	std::vector<int>      labels;
};

class YOLOV5
{
    public:
        // Take a model path as string
        void loadModel(std::string path);
        // Take an image and return a prediction
        void run(cv::Mat image, Prediction &out_pred);

        void getLabelsName(std::string path, std::vector<std::string> &labelNames);

        float confThreshold, nmsThreshold;
        
    private:
        tensorflow::SavedModelBundle bundle;
        tensorflow::SessionOptions session_options;
        tensorflow::RunOptions run_options;

        tensorflow::Tensor preprocess(cv::Mat &image,int &size);
        void nonMaximumSupprition(
                                std::vector<std::vector<float>> &predV,
								std::vector<int> &predSize,
								std::vector<cv::Rect> &boxes,
								std::vector<float> &confidences,
								std::vector<int> &classIds,
								std::vector<int> &indices,
								cv::Size &size);

        std::vector<std::vector<float>> tensorToVector2D(tensorflow::Tensor &tensor, int &row, int &colum);
        std::vector<int> getTensorShape(tensorflow::Tensor &tensor);
        void equalizeIntensity(cv::Mat& image);
        cv::Mat equalizeIntensityYCrCB(const cv::Mat& image);
};
