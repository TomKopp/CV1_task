#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Utils.h"
#include "HarrisDetector.h"



auto lowerThan = [](const float x) {
  return [&](const float y)->bool {
    return y < x;
  };
};

auto greaterThan = [](const float x) {
  return [&](const float y)->bool {
    return y > x;
  };
};

auto between = [](const float x, const float z) {
  return [&](const float y)->bool {
    return (x < y) && (y < z);
  };
};

auto isOverNineThousand = greaterThan(9000.0f);
auto isLowerNineThousand = lowerThan(9000.0f);
auto isBetweenNegOneAndOne = between(-1.0f, 1.0f);

int main(int argc, char** argv) {
  cv::Mat ImgOrig,
    ImgHarris;
  

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

  HarrisDetector Harris(ImgOrig);


  // Create a windows for display
  cv::namedWindow("Original");
  //cv::namedWindow("Sobel");
  cv::namedWindow("Harris");

  // Display Images
  cv::imshow("Original", ImgOrig);
  //cv::imshow("Sobel", Utils::convolveMatWithSobel(ImgResult));
  //cv::imshow("Harris", Harris.filterImgByResponse(between(-30, 30)));
  cv::imshow("Harris", Harris.filterImgByResponse(lowerThan(0)));
  
  std::array<cv::Mat, 3> Derivs = Harris.getDerivatives();
  cv::imshow("DerivatesIx", Derivs[0]);
  cv::imshow("DerivatesIy", Derivs[1]);
  cv::imshow("DerivatesIxy", Derivs[2]);
  //std::cout << Harris.getResponse() << std::endl;

  cv::waitKey();
  return 0;
}