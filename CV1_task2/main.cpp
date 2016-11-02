#include <iostream>

int main(int argc, char** argv) {
  cv::Mat ImgOrig;
  // check if image path is supplied as argument
  if (argc < 2) {
    std::cout << "Path must be applied as commandline argument." << std::endl;
    return -1;
  }

  // read image and check if successful
  ImgOrig = cv::imread(argv[1]);
  if (ImgOrig.empty()) {
    std::cout << "Could not open or find the image." << std::endl;
    return -1;
  }


  // create a window for display
  cv::namedWindow("Original");
  cv::namedWindow("Result");

  cv::waitKey();
  return 0;
}