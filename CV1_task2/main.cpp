#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Utils.h"
#include "HarrisDetector.h"


static bool isLowerZero(float val) {
  return val < 0;
}

static bool isGreaterOneHundreth(float val) {
  return val > .01;
}

static bool isLowerNeg100k(float val) {
  return val < -100000;
}

static bool isGreaterNegOneLowerZero(float val) {
  return (val > -1.0 && val <= 0.0);
}

int main(int argc, char** argv) {
  cv::Mat ImgOrig
    , ImgResult
    , ImgExpMask
    , ImgHarris;
  

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

  //ImgResult = Utils::convertImgToGray(ImgOrig);
  //ImgHarris = cv::imread(argv[2]);
  HarrisDetector Harris(ImgOrig);


  // Create a windows for display
  cv::namedWindow("Original");
  //cv::namedWindow("Sobel");
  cv::namedWindow("Harris");

  // Display Images
  cv::imshow("Original", ImgOrig);
  //cv::imshow("Sobel", Utils::convolveMatWithSobel(ImgResult));
  cv::imshow("Harris", Harris.filterImgByResponses(isGreaterNegOneLowerZero));
  Harris.getDerivatives().Ix.convertTo(ImgHarris, CV_8U);
  cv::imshow("DerivatesIx", ImgHarris);
  Harris.getDerivatives().Iy.convertTo(ImgHarris, CV_8U);
  cv::imshow("DerivatesIy", ImgHarris);
  Harris.getDerivatives().Ixy.convertTo(ImgHarris, CV_8U);
  cv::imshow("DerivatesIxy", ImgHarris);
  //std::cout << Harris.getResponses() << std::endl;

  cv::waitKey();
  return 0;
}