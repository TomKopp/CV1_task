#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Utils.h"

int main(int argc, char** argv) {
  cv::Mat ImgOrig, ImgResult;

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

  // Create a windows for display
  cv::namedWindow("Original");
  cv::namedWindow("Result");

  // Display Images
  cv::imshow("Original", ImgOrig);
  cv::imshow("Result", ImgResult);

  cv::waitKey();
  return 0;
}