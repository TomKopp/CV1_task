#include "HarrisDetector.h"
#include "Utils.h"

#include <map>
#include <opencv2/imgproc/imgproc.hpp>



HarrisDetector::Derivatives HarrisDetector::_computeDerivatives(const cv::Mat_<float> & Img)
{
  Derivatives Derivs;

  // Apply horizontal Sobel kernel to image
  Derivs.Ix = _convolveKernel(Img, (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1));
  // Apply vertical Sobel kernel to image
  Derivs.Iy = _convolveKernel(Img, (cv::Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1));

  Derivs.Ixy = cv::Mat_<float>(Img.size(), 0.0);
  for (int r = 0; r < Derivs.Ix.rows; ++r) {
    for (int c = 0; c < Derivs.Ix.cols; ++c) {
      Derivs.Ixy.at<float>(r, c) = Derivs.Ix.at<float>(r, c) * Derivs.Iy.at<float>(r, c);
    }
  }

  return Derivs;
}

HarrisDetector::Derivatives HarrisDetector::_convolveGaussian(const Derivatives & Derivs)
{
  Derivatives Ret;
  cv::Mat GaussianKernel = (cv::Mat_<float>(5, 5) <<
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 16, 41, 16, 7,
    4, 16, 26, 16, 4, 
    1,  4,  7,  4, 1);
  GaussianKernel = GaussianKernel / 273.0;

  Ret.Ix = _convolveKernel(Derivs.Ix, GaussianKernel);
  Ret.Iy = _convolveKernel(Derivs.Iy, GaussianKernel);
  Ret.Ixy = _convolveKernel(Derivs.Ixy, GaussianKernel);

  return Ret;
}

cv::Mat HarrisDetector::_convolveKernel(const cv::Mat_<float> & Img, const cv::Mat & Kernel)
{
  cv::Mat_<float> Ret(Img.size(), 0.0);

  cv::filter2D(Img, Ret, CV_32F, Kernel);

  return Ret;
}

cv::Mat HarrisDetector::_computeResponse(const Derivatives & Derivs)
{
  cv::Mat_<float> Ret(Derivs.Ix.size(), 0.0);
  // M = |A C| - Structure tensor - second moment matrix - autocorrelation matrix
  //     |C B|
  float M[3] = {};
  float k = 0.04f; // empirical constant: k = 0.04 - 0.06
  float det; // A*B - C^2
  float trace; // A + B
  // float R = det(M) - k * Trace(M)^2 // Response

  for (int r = 0; r < Ret.rows; ++r) {
    for (int c = 0; c < Ret.cols; ++c) {
      // Build structor tensor
      M[0] = Derivs.Ix.at<float>(r, c) * Derivs.Ix.at<float>(r, c); // A = Ix^2
      M[1] = Derivs.Iy.at<float>(r, c) * Derivs.Iy.at<float>(r, c); // B = Iy^2
      M[2] = Derivs.Ix.at<float>(r, c) * Derivs.Iy.at<float>(r, c); // C = Ix * Iy
      // Calculate Determinant of M
      det = M[0] * M[1] - M[2] * M[2];
      // Calculate Trace of M
      trace = M[0] + M[1];
      // calculate Harris response
      Ret.at<float>(r, c) = det - k * trace * trace;
      //Ret.at<float>(r, c) = 2 * det / (trace + 1);
    }
  }

  return Ret;
}

//cv::Mat HarrisDetector::_findLocalMaxima(const cv::Mat_<float>& Img)
//{
//  cv::Mat_<float> Ret(Img.size(), 0.0);
//  float val;
//  std::map<float> myMap;
//
//
//  for (int r = 1; r < Ret.rows - 1; ++r) {
//    for (int c = 1; c < Ret.cols - 1; ++c) {
//      val = Img.at<float>(r, c);
//      if (val < Img.at<float>(r - 1, c - 1)
//        || val < Img.at<float>(r - 1, c)
//        || val < Img.at<float>(r - 1, c + 1)
//        || val < Img.at<float>(r, c - 1)
//        || val < Img.at<float>(r, c + 1)
//        || val < Img.at<float>(r + 1, c - 1)
//        || val < Img.at<float>(r + 1, c)
//        || val < Img.at<float>(r + 1, c + 1)
//        ) {
//        Ret.at<float>(r, c) = 0.0;
//      }
//      else {
//        Ret.at<float>(r, c) = val;
//      }
//    }
//  }
//
//  return Ret;
//}

HarrisDetector::HarrisDetector()
{
}

HarrisDetector::HarrisDetector(const cv::Mat & Img)
{
  _ImgOrig = Img.clone();

  _derivatives = _convolveGaussian(
    _computeDerivatives(
      Utils::convertImgToGray(_ImgOrig)
    )
  );

  _Responses = _computeResponse(
    _convolveGaussian(
      _computeDerivatives(
        Utils::convertImgToGray(_ImgOrig)
      )
    )
  );
}


HarrisDetector::~HarrisDetector()
{
}

cv::Mat HarrisDetector::getResponses()
{
  return _Responses;
}

HarrisDetector::Derivatives HarrisDetector::getDerivatives()
{
  return _derivatives;
}

cv::Mat HarrisDetector::filterImgByResponses(bool(*cmpFnc)(float))
{
  cv::Mat Ret = _ImgOrig.clone();
  Ret.convertTo(Ret, CV_32F);

  // If response value greater/lower/... than a threshold, determinded by the compare function,
  // the value of the original Image is keept else it's blacked out
  for (int r = 0; r < Ret.rows; ++r) {
    for (int c = 0; c < Ret.cols; ++c) {
      if (!cmpFnc(_Responses.at<float>(r, c))) {
        Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0,0,0);
      }
    }
  }

  //Ret.forEach<float>([&](cv::Point &p, const int * position)->void {
  //});

  Ret.convertTo(Ret, CV_8U);
  return Ret;
}
