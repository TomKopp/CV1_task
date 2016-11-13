#pragma once

#include <opencv2/core/core.hpp>


class HarrisDetector
{
private:
  struct Derivatives
  {
    cv::Mat_<float> Ix;
    cv::Mat_<float> Iy;
    cv::Mat_<float> Ixy;
  };

  Derivatives _derivatives;

private:
  Derivatives _calculateDerivatives(const cv::Mat_<float> & Img);
  cv::Mat _convolveKernel(const cv::Mat_<float> & Img, const cv::Mat & kernel);
  // Structure tensor - second moment matrix - autocorrelation matrix
  //cv::Mat _buildStructureTensor(const Derivatives & Derivs);
  cv::Mat _computeResponse(const Derivatives & Derivs);


public:
  HarrisDetector();
  ~HarrisDetector();
};
