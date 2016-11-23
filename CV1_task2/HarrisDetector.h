#pragma once

#include <array>
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

public:
  HarrisDetector();
  HarrisDetector(const cv::Mat & Img);
  ~HarrisDetector();

  cv::Mat_<float> getResponse();
  std::array<cv::Mat, 3> getDerivatives(bool raw = false);

/// <summary>
/// Filters the img by responses.
/// If response value greater/lower/... than a threshold, determinded by the compare function,
/// the value of the original Image is keept else it's blacked out.
/// The compare function gets the float value of the Harris response and has to return a boolean.
/// </summary>
/// <param name="cmpFnc">The compare function.</param>
/// <returns>cv::Mat</returns>
  template<typename Functor> inline
    cv::Mat filterImgByResponse(const Functor& cmpFnc)
  {
    cv::Mat Ret = _ImgOrig.clone();
    Ret.convertTo(Ret, CV_32F);

    for (int r = 0; r < Ret.rows; ++r) {
      for (int c = 0; c < Ret.cols; ++c) {
        if (!cmpFnc(_Response.at<float>(r, c))) {
          Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
        }
        //if (cmpFnc(_Response.at<float>(r, c))) {
        //  Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 255);
        //}
        //else {
        //  Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
        //}
      }
    }

    Ret.convertTo(Ret, _ImgOrig.type());
    return Ret;
  }
};
