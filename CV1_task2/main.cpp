#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Utils.h"



static cv::Mat applySobelX(const cv::Mat& Img) {
  // Accept only char type matrices
  CV_Assert(Img.depth() == CV_8U);

  cv::Mat Res;
  const int nChannels = Img.channels();
  int scan_start_x = 0
    , scan_start_y = 0
    , scan_end_x = (Img.cols - 1)
    , scan_end_y = (Img.rows - 1);
  uint val_nw, val_ne, val_e, val_se, val_sw, val_w, r,l,m;
  uchar *my_point
    , *ptr_row
    , *ptr_row_prev
    , *ptr_row_next;

  Res.create(Img.size(), Img.type());

  //if (Img.isContinuous()) {
  //  ptr_row_prev = Img.data;
  //  ptr_row = ptr_row_prev * Img.cols * Img.channels();
  //}



  //while (scan_start_y < scan_end_y)
  //{
  //  const uchar *ptr_row_prev = Img.ptr<uchar>(scan_start_y);
  //  const uchar *ptr_row = Img.ptr<uchar>(++scan_start_y);
  //  const uchar *ptr_row_next = Img.ptr<uchar>(scan_start_y + 1);

  //  uchar *ptr_out = Res.ptr<uchar>(scan_start_y);

  //  while (scan_start_x < scan_end_x)
  //  {
  //    r = scan_start_x;
  //    m = ++scan_start_x;
  //    l = m + 1;
  //    val_nw = ptr_row_prev[r];
  //    val_w = ptr_row[r];
  //    val_sw = ptr_row_next[r];

  //    //my_point = &ptr_row[m];

  //    val_ne = ptr_row_prev[l];
  //    val_e = ptr_row[l];
  //    val_se = ptr_row_next[l];

  //    // Confused yet? Fear not, it will get worse.
  //    //*my_point = val_ne + (val_e << 1) + val_se - val_nw - (val_w << 1) - val_sw;
  //    ptr_out[m] = 222;
  //  }
  //}

  while (scan_start_y < scan_end_y)
  {
    ptr_row = Res.ptr<uchar>(++scan_start_y);

    while (scan_start_x < scan_end_x)
    {
      my_point = &ptr_row[ (++scan_start_x * nChannels)];
      *my_point = 222;
    }
  }

  //for (scan_start_y; scan_start_y < scan_end_y; scan_start_y++)
  //{
  //  uchar *ptr_out = Res.ptr<uchar>(scan_start_y);

  //  for (scan_start_x; scan_start_x < scan_end_x; scan_start_x++)
  //  {
  //    my_point = &ptr_out[scan_start_x];
  //    *my_point = 222;
  //  }
  //}

  return Res;
}

static cv::Mat applySobel(const cv::Mat& Img) {
  cv::Mat Res;
  Res.create(Img.size(), Img.type());

  cv::Mat kern = (cv::Mat_<char>(3, 3) << -1, 0, 1,   -2, 0, 2,   -1, 0, 1);

  cv::filter2D(Img, Res, Img.depth(), kern);

  return Res;
}



int main(int argc, char** argv) {
  cv::Mat ImgOrig, ImgResult;

  // Check if image path is supplied as argument
  if (argc < 2) {
    std::cout << "Path must be applied as commandline argument." << std::endl;
    return -1;
  }

  // Read image and check if successful
  ImgOrig = cv::imread(argv[1]);
  if (ImgOrig.empty()) {
    std::cout << "Could not open or find the image." << std::endl;
    return -1;
  }

  ImgResult = Utils::convertImgToGray(ImgOrig);

  // Create a windows for display
  cv::namedWindow("Original");
  cv::namedWindow("Result");

  // Display Images
  cv::imshow("Original", ImgOrig);
  cv::imshow("Result", applySobelX(ImgResult));

  cv::waitKey();
  return 0;
}