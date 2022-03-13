#include "yolov5.h"
#include <fstream>

void YOLOV5::loadModel(std::string path){
    auto status = tensorflow::LoadSavedModel(session_options, run_options, path, {"serve"}, &bundle);
	if (status.ok())
	{
		printf("\n\n\nModel loaded successfully...\n\n\n");
	}
	else
	{
		printf("\n\n\nError in loading model\n\n\n");
	}
}

void YOLOV5::run(cv::Mat image, Prediction &out_pred){  
    std::vector<int> indices;
	std::vector<int>  classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect>  boxes;
    cv::Size size = image.size();
	int reSizeTo = 640;

    // resize, normalize and convert to tensor 
    tensorflow::Tensor img  = preprocess(image,reSizeTo);

    // model config
    const std::string input_node = "serving_default_input_1:0"; 
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_data = {{input_node, img}};
	std::vector<std::string> output_nodes = {"StatefulPartitionedCall:0"};

	std::vector<tensorflow::Tensor> predictions;

    // predict 
	bundle.GetSession()->Run(inputs_data, output_nodes, {}, &predictions);
    tensorflow::Tensor pred = predictions[0].SubSlice(0);

	// std::cout << pred.DebugString() << std::endl;

    // convert tensor to vector
    std::vector<int> predSize = getTensorShape(pred);
	std::vector<std::vector<float>> predV = tensorToVector2D(pred, predSize[0], predSize[1]);

    nonMaximumSupprition(predV,predSize,boxes,confidences,classIds,indices,size);

	for(int i =0 ; i < indices.size() ; i++ ){
		out_pred.boxes.push_back(boxes[indices[i]]);
		out_pred.scores.push_back(confidences[indices[i]]);
		out_pred.labels.push_back(classIds[indices[i]]);
	}


}

tensorflow::Tensor YOLOV5::preprocess(cv::Mat &image,int &size){
    	
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	//equalizeIntensity(image);   // GRAY
	//image = equalizeIntensityYCrCB(image); // YCrCb
	cv::resize(image, image, cv::Size(size, size));
	tensorflow::Tensor tensorImage(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, size, size, 3}));//3
	float *p = tensorImage.flat<float>().data();
	cv::Mat outputImg(640, 640, CV_32FC3, p);
	image.convertTo(outputImg, CV_32FC3, 1.0/255);
	return tensorImage;

}

void YOLOV5::nonMaximumSupprition(
								std::vector<std::vector<float>> &predV,
								std::vector<int> &predSize,
								std::vector<cv::Rect> &boxes,
								std::vector<float> &confidences,
								std::vector<int> &classIds,
								std::vector<int> &indices,
								cv::Size &size)
								
{

	std::vector<cv::Rect> boxesNMS;
	int max_wh = 40960;
	std::vector<float> scores;
	double confidence;
	cv::Point classId;

	
    for (int i = 0; i < predSize[0]; i++)
	{
		if ( predV[i][4] > confThreshold )
		{
			// height--> image.rows,  width--> image.cols;
			int left = (predV[i][0] - predV[i][2] / 2) *  size.width;
			int top = (predV[i][1] - predV[i][3] / 2) * size.height;
			int w = predV[i][2] *  size.width;
			int h = predV[i][3] * size.height;


			for (int j = 5; j < predSize[1]; j++)
			{
				// # conf = obj_conf * cls_conf
				scores.push_back(predV[i][j] * predV[i][4]);
			}
			
			cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
			scores.clear();
			int c = classId.x * max_wh ; 
			if (confidence > confThreshold)
			{
				boxes.push_back(cv::Rect(left, top, w, h));
				// std::cout << top << " "<< left << " " << w << " " << h <<std::endl;
				// top =  top + c;
				// left = left + c;
				// w =   w + c;
				// h =   h +c;
				confidences.push_back(confidence);
				classIds.push_back(classId.x);

				boxesNMS.push_back(cv::Rect(left, top, w, h));
			}
		}
	}
	//std::cout << classId.x  << "  " << max_wh << "  " <<classId.x * max_wh << "     classId.x * max_wh" << std::endl;
	cv::dnn::NMSBoxes(boxesNMS, confidences, confThreshold, nmsThreshold, indices);
}

std::vector<std::vector<float>> YOLOV5::tensorToVector2D(tensorflow::Tensor &tensor, int &row, int &colum)
{
	float *tensor_ptr = tensor.flat<float>().data();
	std::vector<std::vector<float>> v;
	for (int i = 0; i < row; ++i)
	{
		std::vector<float> tem(tensor_ptr + (i * colum), tensor_ptr + (colum * (i + 1)));
		v.push_back(tem);
	}
	return v;
}

std::vector<int> YOLOV5::getTensorShape(tensorflow::Tensor &tensor)
{
	std::vector<int> shape;
	int num_dimensions = tensor.shape().dims();
	for (int i = 0; i < num_dimensions; i++)
	{
		shape.push_back(tensor.shape().dim_size(i));
	}
	
	return shape;
}

void YOLOV5::getLabelsName(std::string path, std::vector<std::string> &labelNames)
{
    // Open the File
    std::ifstream in(path.c_str());
    // Check if object is valid
    if(!in)
        throw std::runtime_error("Can't open ");
        
    std::string str;
    // Read the next line from File until it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            labelNames.push_back(str);
    }
    //Close The File
    in.close();
}


void YOLOV5::equalizeIntensity(cv::Mat& image)
{
        cv::Mat gray;
        cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray,gray);
        cv::cvtColor(gray,image,cv::COLOR_GRAY2BGR);
}

cv::Mat YOLOV5::equalizeIntensityYCrCB(const cv::Mat& image) // light intensity
{
    if(image.channels() >= 3)
    {
        cv::Mat ycrcb;
        cv::cvtColor(image,ycrcb,cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb,channels);
        cv::equalizeHist(channels[0], channels[0]);
        cv::Mat result;
        cv::merge(channels,ycrcb);
        cv::cvtColor(ycrcb,result,cv::COLOR_YCrCb2BGR);
        return result;
    }
    return cv::Mat();
}