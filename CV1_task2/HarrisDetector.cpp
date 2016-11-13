#include "HarrisDetector.h"
#include "Utils.h"

#include <opencv2/imgproc/imgproc.hpp>



HarrisDetector::Derivatives HarrisDetector::_calculateDerivatives(const cv::Mat_<float> & Img)
{
  Derivatives Derivs;

  // Apply horizontal Sobel kernel to image
  Derivs.Ix = _convolveKernel(Img, (cv::Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1));
  // Apply vertical Sobel kernel to image
  Derivs.Iy = _convolveKernel(Img, (cv::Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1));

  for (int r = 0; r < Derivs.Ix.rows; ++r) {
    for (int c = 0; c < Derivs.Ix.cols; ++c) {
      Derivs.Ixy.at<float>(r, c) = Derivs.Ix.at<float>(r, c) * Derivs.Iy.at<float>(r, c);
    }
  }

  return Derivs;
}

cv::Mat HarrisDetector::_convolveKernel(const cv::Mat_<float> & Img, const cv::Mat & Kernel)
{
  cv::Mat_<float> Res(Img.size(), 0.0);

  cv::filter2D(Img, Res, Img.depth(), Kernel);

  return Res;
}

cv::Mat HarrisDetector::_computeResponse(const Derivatives & Derivs)
{
  cv::Mat_<float> Res(Derivs.Ix.size(), 0.0);
  // Structure tensor M = |A C|
  //                      |C B|
  float M[3] = {}; 
  float k = 0.04; // empirical constant: k = 0.04 - 0.06
  float det; // A*B - C^2
  float trace; // A + B
  // float R = det(M) - k * Trace(M)^2 // Response

  for (int r = 0; r < Res.rows; ++r) {
    for (int c = 0; c < Res.cols; ++c) {
      M[0] = Derivs.Ix.at<float>(r, c) * Derivs.Ix.at<float>(r, c);
      M[1] = Derivs.Iy.at<float>(r, c) * Derivs.Iy.at<float>(r, c);
      M[2] = Derivs.Ix.at<float>(r, c) * Derivs.Iy.at<float>(r, c);
      det = M[0] * M[1] - M[2] * M[2];
      trace = M[0] + M[1];

      Res.at<float>(r, c) = det - k * trace * trace;
    }
  }

  return Res;
}

HarrisDetector::HarrisDetector()
{
}


HarrisDetector::~HarrisDetector()
{
}
