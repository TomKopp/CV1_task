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
  : _ImgOrig(Img.clone())
{
  cv::Mat WorkingCopy;
  std::array<cv::Mat, 3> StructureTensor;
  _ImgOrig.convertTo(WorkingCopy, CV_32F);
  WorkingCopy = Utils::convertImgToGray(WorkingCopy);

  _Derivatives = _computeDerivatives(WorkingCopy);
  
  StructureTensor[0] = _convolveGaussian(_Derivatives[0].mul(_Derivatives[0])); // A = X^2 * w
  StructureTensor[1] = _convolveGaussian(_Derivatives[1].mul(_Derivatives[1])); // B = Y^2 * w
  StructureTensor[2] = _convolveGaussian(_Derivatives[2]); // C = (XY) * w

  _Response = _computeResponse(StructureTensor);
}

/// <summary>
/// Finalizes an instance of the <see cref="HarrisDetector"/> class.
/// </summary>
HarrisDetector::~HarrisDetector()
{
}

/// <summary>
/// Convolves the image with the kernel.
/// </summary>
/// <param name="Img">The img.</param>
/// <param name="Kernel">The kernel.</param>
/// <returns>cv::Mat</returns>
cv::Mat HarrisDetector::_convolveKernel(const cv::Mat & Img, const cv::Mat & Kernel)
{
  cv::Mat Ret(Img.size(), CV_32F, cv::Scalar(0.0));

  cv::filter2D(Img, Ret, CV_32F, Kernel);

  return Ret;
}

/// <summary>
/// Convolves the image with a gaussian kernel.
/// </summary>
/// <param name="Img">The img.</param>
/// <returns>cv::Mat</returns>
cv::Mat HarrisDetector::_convolveGaussian(const cv::Mat & Img)
{
  cv::Mat GaussianKernel = (cv::Mat_<float>(5, 5) <<
    1, 4, 7, 4, 1,
    4, 16, 26, 16, 4,
    7, 16, 41, 16, 7,
    4, 16, 26, 16, 4,
    1, 4, 7, 4, 1);
  GaussianKernel = GaussianKernel / 273.0;

  return _convolveKernel(Img, GaussianKernel);
}

/// <summary>
/// Computes the derivatives.
/// </summary>
/// <param name="Img">The img.</param>
/// <returns>std::array</returns>
std::array<cv::Mat, 3> HarrisDetector::_computeDerivatives(const cv::Mat & Img)
{
  std::array<cv::Mat, 3> Ret;

  Ret[0] = _convolveKernel(Img, (cv::Mat_<float>(1, 3) << -1, 0, 1)); // X = I * (-1, 0, 1)
  Ret[1] = _convolveKernel(Img, (cv::Mat_<float>(3, 1) << -1, 0, 1)); // Y = I * (-1, 0, 1)T
  Ret[2] = Ret[0].mul(Ret[1]); // XY

  return Ret;
}

/// <summary>
/// Computes the Harris response for each element.
/// All Structure tensor elements must have the same size.
/// </summary>
/// <param name="StructureTensor">The structure tensor.</param>
/// <returns>cv::Mat</returns>
cv::Mat HarrisDetector::_computeResponse(const std::array<cv::Mat, 3>& StructureTensor)
{
  CV_Assert(
    StructureTensor[0].size() == StructureTensor[1].size()
    && StructureTensor[0].size() == StructureTensor[2].size()
  );

  cv::Mat
    Ret(StructureTensor[0].size(), CV_32F, cv::Scalar(0.0)),
    A = StructureTensor[0],
    B = StructureTensor[1],
    C = StructureTensor[2];
  float
    k = 0.04f, // empirical constant: k = 0.04 - 0.06
    det, // Det = AB - C^2
    tr; // Tr = A + B

  for (int r = 0; r < Ret.rows; r++) {
    for (int c = 0; c < Ret.cols; c++) {
      det = A.at<float>(r, c) * B.at<float>(r, c) - C.at<float>(r, c) * C.at<float>(r, c); // Det = AB - C^2
      tr = A.at<float>(r, c) + B.at<float>(r, c); // Tr = A + B
      Ret.at<float>(r, c) = det - k * tr  * tr; // R = Det - k * Tr^2
    }
  }

  return Ret;
}

/// <summary>
/// Performs the non-maxima suppression on a given 3x3 neighboorhood.
/// </summary>
/// <param name="Response">The response.</param>
/// <returns></returns>
cv::Mat HarrisDetector::_nonMaximaSuppression(const cv::Mat & Response)
{
  cv::Mat Ret(Response.size(), CV_32F, cv::Scalar(0.0)); // == Mask
  int
    c = 2,
    r = 2,
    h = Ret.rows,
    w = Ret.cols,
    cur = 1,
    next = 2;
  float val; // current pixels value

             // USE EIGEN!!!!
  bool(*skip)[2] = new bool[h][2](); // scanline masks
  for (int i = 0; i < h; i++) {
    skip[i][0] = false;
    skip[i][1] = false;
  }

  for (c; c < w - 1; c++) {
    r = 2; // set r every start of the loop to two

    while (r < h) {
      if (skip[r][cur]) { // skip current pixel
        r++;
        continue;
      }
      val = Response.at<float>(r, c);

      if (val <= Response.at<float>(r + 1, c)) { // compare to pixel on the left
        r++;
        while (r < h && val <= Response.at<float>(r + 1, c)) { // rising
          r++;
        }
        if (r == h) { // reach scanline's local maximum
          break;
        }
      }
      else {
        if (val <= Response.at<float>(r - 1, c)) { // compare to pixel on the right
          r++;
          continue;
        }
      }
      skip[r + 1][cur] = true; // skip next pixel in the scanline

                               // compare to 3 future then 3 past neighbors
      if (val <= Response.at<float>(r - 1, c + 1)) {
        r++;
        continue;
      }
      skip[r - 1][next] = true; // skip future neighbors only

      if (val <= Response.at<float>(r, c + 1)) {
        r++;
        continue;
      }
      skip[r][next] = true;

      if (val <= Response.at<float>(r + 1, c + 1)) {
        r++;
        continue;
      }
      skip[r + 1][next] = true;

      if (val <= Response.at<float>(r - 1, c - 1)) { r++; continue; }
      if (val <= Response.at<float>(r, c - 1)) { r++; continue; }
      if (val <= Response.at<float>(r + 1, c - 1)) { r++; continue; }

      Ret.at<float>(r, c) = Response.at<float>(r, c); // a new local maximum is found
    }

    // swap skip mask indices
    std::swap(cur, next);
    for (int i = 0; i < h; i++) { // reset next scanline mask
      skip[i][next] = false;
    }
  }

  for (int i = 0; i < h; i++) { // cleanup scanline mask
    delete skip[i];
  }
  delete skip;
  return Ret;
}

/// <summary>
/// Performs the non-maxima suppression on a given neighboorhood size.
/// </summary>
/// <param name="Response">The response.</param>
/// <param name="Neighborhood">The neighborhood defined as (2n + 1)×(2n + 1). Give n</param>
/// <returns></returns>
cv::Mat HarrisDetector::_nonMaximaSuppression(const cv::Mat & Response, uchar Neighborhood)
{

  return cv::Mat();
}

/// <summary>
/// Gets the Harris response.
/// </summary>
/// <returns>cv::Mat</returns>
cv::Mat HarrisDetector::getResponse()
{
  cv::Mat Ret = _Response.clone();
  return Ret;
}

/// <summary>
/// Gets the derivatives.
/// </summary>
/// <param name="raw">if set to <c>true</c> [raw].</param>
/// <returns>std::array</returns>
std::array<cv::Mat, 3> HarrisDetector::getDerivatives(bool raw)
{
  std::array<cv::Mat, 3> Ret = {
    _Derivatives[0].clone(),
    _Derivatives[1].clone(),
    _Derivatives[2].clone()
  };

  if (!raw) {
    Ret[0].convertTo(Ret[0], _ImgOrig.type());
    Ret[1].convertTo(Ret[1], _ImgOrig.type());
    Ret[2].convertTo(Ret[2], _ImgOrig.type());
  }

  return Ret;
}
