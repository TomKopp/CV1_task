#pragma once

#include <array>
#include <opencv2/core/core.hpp>


class HarrisDetector
{
private:
  cv::Mat _ImgOrig;
  cv::Mat _Response;
  std::array<cv::Mat, 3> _Derivatives;

private:
  cv::Mat _convolveKernel(const cv::Mat & Img, const cv::Mat & Kernel);
  cv::Mat _convolveGaussian(const cv::Mat & Img);
  std::array<cv::Mat, 3> _computeDerivatives(const cv::Mat & Img);
  cv::Mat _computeResponse(const std::array<cv::Mat, 3> & StructureTensor);
  cv::Mat _nonMaximaSuppression(const cv::Mat & Response);
  cv::Mat _nonMaximaSuppression(const cv::Mat & Response, uchar Neighborhood = 1);

public:
  HarrisDetector();
  HarrisDetector(const cv::Mat & Img);
  ~HarrisDetector();

  cv::Mat getResponse();
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

    for (int r = 0; r < Ret.rows; r++) {
      for (int c = 0; c < Ret.cols; c++) {
        //if (!cmpFnc(_Response.at<float>(r, c))) {
        //  Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
        //}
        if (cmpFnc(_Response.at<float>(r, c))) {
          Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 255);
        }
        else {
          Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
        }
      }
    }

    Ret.convertTo(Ret, _ImgOrig.type());
    return Ret;
  }
};
