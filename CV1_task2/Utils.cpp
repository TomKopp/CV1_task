#include "Utils.h"

#include <opencv2/imgproc/imgproc.hpp>


Utils::Utils()
{
}


Utils::~Utils()
{
}

cv::Mat Utils::convertImgToGray(const cv::Mat & Img)
{
  cv::Mat ret = cv::Mat(Img);
  cvtColor(ret, ret, cv::COLOR_BGR2GRAY);

  return ret;
}

cv::Mat Utils::convertImgToBGR(const cv::Mat & Img)
{
  cv::Mat ret = cv::Mat(Img);
  cvtColor(ret, ret, cv::COLOR_GRAY2BGR);

  return ret;
}

bool Utils::isPointOffMat(const cv::Mat & Img, const cv::Point & Pnt)
{
  return (Pnt.x < 0 || Pnt.y < 0 || Pnt.x < Img.cols - 1 || Pnt.y > Img.rows - 1);
}

bool Utils::isPointWithinMat(const cv::Mat & Img, const cv::Point & Pnt)
{
  return !isPointOffMat(Img, Pnt);
}

void Utils::drawPointInMat(cv::Mat & Img, const cv::Point & Pnt, const cv::Vec3b & Color = cv::Vec3b(0,0,0))
{
  Img.at<cv::Vec3b>(Pnt) = Color;
}

void Utils::drawPointInMat(cv::Mat & Img, const cv::Point & Pnt, uchar Color)
{
  Img.at<cv::Vec3b>(Pnt) = cv::Vec3b(Color, Color, Color);
}

