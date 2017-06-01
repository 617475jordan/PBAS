#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "pbas.h"
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;
	/* Open video file */
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		std::cerr << "Cannot open video!" << std::endl;
		return 1;
	}
	PBAS pbas;
	std::cout << "Press 'q' to quit..." << std::endl;
	int key = 0;
	Mat img_input;
	capture >> img_input;
	while (key != 'q')
	{
		capture >> img_input;
		cv::imshow("Input", img_input);
		cv::GaussianBlur(img_input, img_input, cv::Size(5, 5), 1.5);
		cv::Mat img_mask;
		double t = (double)cv::getTickCount();
		pbas.process(&img_input, &img_mask);
		t = (double)cv::getTickCount() - t;
		int fps = (int)(cv::getTickFrequency() / t);
		std::cout << fps << std::endl;
		cv::medianBlur(img_mask, img_mask, 5);
		imshow("img_mask", img_mask);

		key = cvWaitKey(1);
	}
	return 0;
}﻿﻿