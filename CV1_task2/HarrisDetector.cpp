#include "HarrisDetector.h"
#include "Utils.h"

#include <opencv2/imgproc/imgproc.hpp>



/// <summary>
/// Initializes a new instance of the <see cref="HarrisDetector"/> class.
/// </summary>
HarrisDetector::HarrisDetector()
{
}

/// <summary>
/// Initializes a new instance of the <see cref="HarrisDetector"/> class.
/// Calculates the Harris corner response.
/// 
/// A Combined Corner And Edge Detector - Harris & Stephens
/// X = I * (-1, 0, 1)
/// Y = I * (-1, 0, 1)T
/// E(x,y) = A(x^2) + Cxy + B(y^2)
/// A = X^2 * w
/// B = Y^2 * w
/// C = (XY) * w
/// w_u,v = exp -(u^2 + v^2) / 2 * sigma^2 -> Gaussian kernel
/// E(x,y) = (x, y) M (x, y)T
/// M = |A C|  // Structure tensor - second moment matrix - autocorrelation matrix
///     |C B|
/// alpha, beta = eigenvalues of M
/// Tr(M) = alpha + beta = A + B // Trace
/// Det(M) = alpha*beta = AB - C^2 // Determinant
/// Corner response:
/// R = Det - k * Tr^2
/// R is positive in the corner region, negative in the edge region and small in the flat region
/// The flat region is specified by Tr falling below some selected threshold
/// Corner region pixel is an 8-way local maximum
/// Edge region pixel if R negative and local minimum
/// </summary>
/// <param name="Img">The img.</param>
HarrisDetector::HarrisDetector(const cv::Mat & Img)
{
  _ImgOrig = Img.clone();
  _Derivatives = _computeDerivatives(Utils::convertImgToGray(_ImgOrig));

  // A = X^2 * w
  _StructureTensorElements[0] = _convolveGaussian(_Derivatives.Ix.mul(_Derivatives.Ix));
  // B = Y^2 * w
  _StructureTensorElements[1] = _convolveGaussian(_Derivatives.Iy.mul(_Derivatives.Iy));
  // C = (XY) * w
  _StructureTensorElements[2] = _convolveGaussian(_Derivatives.Ixy);

  _Response = _computeResponse(_StructureTensorElements);
}

/// <summary>
/// Finalizes an instance of the <see cref="HarrisDetector"/> class.
/// </summary>
HarrisDetector::~HarrisDetector()
{
}

/// <summary>
/// Computes the derivatives.
/// </summary>
/// <param name="Img">The img.</param>
/// <returns>Derivatives</returns>
HarrisDetector::Derivatives HarrisDetector::_computeDerivatives(const cv::Mat_<float> & Img)
{
  Derivatives Ret;

  // X = I * (-1, 0, 1)
  Ret.Ix = _convolveKernel(Img, (cv::Mat_<float>(1, 3) << -1, 0, 1));
  // Y = I * (-1, 0, 1)T
  Ret.Iy = _convolveKernel(Img, (cv::Mat_<float>(3, 1) << -1, 0, 1));
  // XY
  Ret.Ixy = Ret.Ix.mul(Ret.Iy);

  return Ret;
}

/// <summary>
/// Convolves the Img with Gaussian kernel.
/// Sigma = 1
/// </summary>
/// <param name="Img">The img.</param>
/// <returns>cv::Mat_</returns>
cv::Mat_<float> HarrisDetector::_convolveGaussian(const cv::Mat_<float> & Img)
{
  cv::Mat GaussianKernel = (cv::Mat_<float>(5, 5) <<
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 16, 41, 16, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1);
  GaussianKernel = GaussianKernel / 273.0;

  return _convolveKernel(Img, GaussianKernel);
}

/// <summary>
/// Convolves the Img with the kernel.
/// </summary>
/// <param name="Img">The img.</param>
/// <param name="Kernel">The kernel.</param>
/// <returns>cv::Mat_</returns>
cv::Mat_<float> HarrisDetector::_convolveKernel(const cv::Mat_<float> & Img, const cv::Mat & Kernel)
{
  cv::Mat_<float> Ret(Img.size(), 0.0);

  cv::filter2D(Img, Ret, CV_32F, Kernel);

  return Ret;
}

/// <summary>
/// Computes the Harris response for each element.
/// All StructureTensorElements must have the same size.
/// </summary>
/// <param name="StructureTensorElements">The structure tensor elements.</param>
/// <returns>cv::Mat_</returns>
cv::Mat_<float> HarrisDetector::_computeResponse(const cv::Mat_<float> StructureTensorElements[])
{
  CV_Assert(StructureTensorElements[0].size() == StructureTensorElements[1].size()
    && StructureTensorElements[0].size() == StructureTensorElements[2].size());

  cv::Mat_<float> Ret(StructureTensorElements[0].size(), 0.0);
  cv::Mat_<float> A = StructureTensorElements[0],
    B = StructureTensorElements[1],
    C = StructureTensorElements[2];
  float k = 0.04f; // empirical constant: k = 0.04 - 0.06
  float det;
  float tr;

  for (int r = 0; r < Ret.rows; r++)
  {
    for (int c = 0; c < Ret.cols; c++)
    {
      // Det = AB - C^2
      det = A.at<float>(r, c) * B.at<float>(r, c) - C.at<float>(r, c) * C.at<float>(r, c);
      // Tr = A + B
      tr = A.at<float>(r, c) + B.at<float>(r, c);
      // R = Det - k * Tr^2
      Ret.at<float>(r, c) = det - k * tr  * tr;
    }
  }

  return Ret;
}

/// <summary>
/// Gets the Harris response.
/// </summary>
/// <returns>cv::Mat</returns>
cv::Mat_<float> HarrisDetector::getResponse()
{
  return _Response;
}

/// <summary>
/// Gets the derivatives.
/// </summary>
/// <returns>std::array<cv::Mat, 3></returns>
std::array<cv::Mat, 3> HarrisDetector::getDerivatives(bool raw)
{
  std::array<cv::Mat, 3> Ret = {
    _Derivatives.Ix.clone(),
    _Derivatives.Iy.clone(),
    _Derivatives.Ixy.clone()
  };

  if (!raw) {
    Ret[0].convertTo(Ret[0], _ImgOrig.type());
    Ret[1].convertTo(Ret[1], _ImgOrig.type());
    Ret[2].convertTo(Ret[2], _ImgOrig.type());
  }

  return Ret;
}
