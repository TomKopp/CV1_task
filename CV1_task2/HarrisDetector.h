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
  cv::Mat _Response;
  Derivatives _Derivatives;
  cv::Mat_<float> _StructureTensorElements[3];

private:
  Derivatives _computeDerivatives(const cv::Mat_<float> & Img);
  cv::Mat_<float> _convolveGaussian(const cv::Mat_<float>& Img);
  cv::Mat_<float> _convolveKernel(const cv::Mat_<float> & Img, const cv::Mat & kernel);
  cv::Mat_<float> _computeResponse(const cv::Mat_<float> StructureTensorElements[]);
  //Derivatives _convolveDerivativesWithGaussian(const Derivatives & Derivs);
  //cv::Mat _computeDerivativesResponse(const Derivatives & Derivs);
  //cv::Mat _findLocalMaxima(const cv::Mat_<float> & Img);

public:
  HarrisDetector();
  HarrisDetector(const cv::Mat & Img);
  ~HarrisDetector();

  cv::Mat_<float> getResponse();
  Derivatives getDerivatives();
  cv::Mat filterImgByResponses(bool(*cmpFnc)(float));
};
