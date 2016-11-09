#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Utils.h"



static cv::Mat applySobel(const cv::Mat& Img) {
  cv::Mat Res;
  Res.create(Img.size(), Img.type());

  cv::Mat kern = (cv::Mat_<char>(3, 3) << -1, 0, 1,   -2, 0, 2,   -1, 0, 1);

  cv::filter2D(Img, Res, Img.depth(), kern);

  return Res;
}



int main(int argc, char** argv) {
  cv::Mat ImgOrig, ImgResult, ImgExpMask;

  // Check if image path is supplied as argument
  if (argc < 2) {
    std::cout << "Path must be applied as commandline argument." << std::endl;
    return -1;
  }

  // Read image and check if successful
  ImgOrig = cv::imread(argv[1]);
  if (ImgOrig.empty()) {
    std::cout << "Could not open or find the image." << std::endl;
    return -1;
  }

  ImgResult = Utils::convertImgToGray(ImgOrig);
  ImgExpMask = cv::Mat(1, 500, CV_8U);

  // Initalize ImgExpMask with random values
  for (int i = 0; i < ImgExpMask.total(); i++) {
    int val_random = rand() % 255;
    ImgExpMask.at<uchar>(i) = cv::saturate_cast<uchar>(val_random);
  }


  // Create a windows for display
  cv::namedWindow("Original");
  cv::namedWindow("Sobel");
  cv::namedWindow("ExpMaskOrig");
  cv::namedWindow("ExpMask");

  // Display Images
  cv::imshow("Original", ImgOrig);
  cv::imshow("Sobel", Utils::convolveMatWithSobel(ImgResult));
  cv::imshow("ExpMaskOrig", ImgExpMask);
  cv::imshow("ExpMask", Utils::convolveMatWithExpMask1D(ImgExpMask));

  cv::waitKey();
  return 0;
}