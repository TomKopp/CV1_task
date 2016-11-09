#pragma once

#include <opencv2/core.hpp>

class Utils
{
public:
  Utils();
  ~Utils();

  /// <summary>
  /// Converts the img to gray.
  /// </summary>
  /// <param name="Img">The img.</param>
  /// <returns>cv::Mat</returns>
  static cv::Mat convertImgToGray(const cv::Mat& Img);

  /// <summary>
  /// Converts the img to BGR.
  /// </summary>
  /// <param name="Img">The img.</param>
  /// <returns>cv::Mat</returns>
  static cv::Mat convertImgToBGR(const cv::Mat& Img);

  /// <summary>
  /// Determines whether Point lies outside the boundaries of the Image.
  /// </summary>
  /// <param name="Img">The Image.</param>
  /// <param name="Pnt">The Point.</param>
  /// <returns>
  ///   <c>true</c> if the Point lies outside the Image; otherwise, <c>false</c>.
  /// </returns>
  static bool isPointOffMat(const cv::Mat& Img, const cv::Point& Pnt);

  /// <summary>
  /// Determines whether Point lies within the boundaries of the Image.
  /// </summary>
  /// <param name="Img">The Image.</param>
  /// <param name="Pnt">The Point.</param>
  /// <returns>
  ///   <c>true</c> if the Point lies within the Image; otherwise, <c>false</c>.
  /// </returns>
  static bool isPointWithinMat(const cv::Mat& Img, const cv::Point& Pnt);

  /// <summary>
  /// Draws the point in mat.
  /// </summary>
  /// <param name="Img">The img.</param>
  /// <param name="Pnt">The PNT.</param>
  /// <param name="Color">The color.</param>
  static void drawPointInMat(cv::Mat& Img, const cv::Point& Pnt, const cv::Vec3b& Color);
  static void drawPointInMat(cv::Mat& Img, const cv::Point& Pnt, uchar Color);

  /// <summary>
  /// Convolves the mat with sobel.
  /// </summary>
  /// <param name="Img">The img.</param>
  /// <returns>cv::Mat</returns>
  static cv::Mat convolveMatWithSobel(const cv::Mat& Img);

  /// <summary>
  /// Convolves the mat with exponentional mask for 1D.
  /// </summary>
  /// <param name="Img">The img.</param>
  /// <returns>cv::Mat</returns>
  static cv::Mat convolveMatWithExpMask1D(const cv::Mat & Img);
};
