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

  cv::Mat _ImgOrig;
  cv::Mat _Responses;

private:
  Derivatives _computeDerivatives(const cv::Mat_<float> & Img);
  Derivatives _convolveGaussian(const Derivatives & Derivs);
  cv::Mat _convolveKernel(const cv::Mat_<float> & Img, const cv::Mat & kernel);
  cv::Mat _computeResponse(const Derivatives & Derivs);

public:
  HarrisDetector();
  HarrisDetector(const cv::Mat & Img);
  ~HarrisDetector();

  cv::Mat getResponses();
  cv::Mat filterImgByResponses(bool(*f)(float));
};
