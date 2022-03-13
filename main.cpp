#include "yolov5.h"


int main(int argc, char* argv[]){
	if (argc != 4){	
		std::cout << "Error! Usage: <path/to_saved_model> <path/to_input/image.jpg> <path/to/output/image.jpg>" << std::endl;
		return 1;
	}

	Prediction out_pred;
	const std::string model_path = argv[1]; 
	const std::string image_path  = argv[2];
	const std::string predicted_image_path = argv[3];
	
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
	// Load the saved_model
	YOLOV5 model;
	
	// conf
	model.confThreshold = 0.40;
	model.nmsThreshold  = 0.40;
	

	model.loadModel(model_path);
	// start 
	auto start = std::chrono::high_resolution_clock::now();	
	// Predict on the input image
	model.run(img, out_pred);
	// calculate the run time and print it
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << duration.count() << std::endl;


	// add the bbox to the image and save it
	auto boxes = out_pred.boxes;
	auto scores = out_pred.scores;
	for (int i=0; i < boxes.size(); i++){
	    auto box = boxes[i];
	    auto score = scores[i];
		cv::rectangle(img, box, cv::Scalar(255, 0, 0), 2);
	}
	cv::cvtColor(img,img, cv::COLOR_BGR2RGB);
	cv::imwrite(predicted_image_path,img);
}
