#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

Point StartPoint = {10, 10};
Point EndPoint = {10, 5};


struct PointHueDistance
{
  Point p;
  int dHue;
};

typedef vector<PointHueDistance> PointsHueDistance;


static bool cmpPointHueDistance(PointHueDistance A, PointHueDistance B) {
  return A.dHue < B.dHue;
}

static Mat convertImgToGray(Mat& ImgOrig) {
  Mat ImgGray;

  ImgGray = ImgOrig.clone();
  cvtColor(ImgGray, ImgGray, COLOR_BGR2GRAY);

  return ImgGray;
}

static int getHueDistance(Mat& Img, Point A, Point B) {
  uchar A_val, B_val, deltaHue;

  A_val = Img.at<uchar>(A);
  B_val = Img.at<uchar>(B);
  deltaHue = A_val - B_val;

  if (deltaHue < 0) {
    deltaHue *= (-1);
  }

  return (int)deltaHue;
}

static Point getNextPix(Mat& Img, Point p, Point last) {
  // all adjacent points with initial hue distance
  //PointsHueDistance adjacentPix = {
  //  { Point(p.x - 1, p.y - 1), INT_MAX },{ Point(p.x, p.y - 1), INT_MAX },{ Point(p.x + 1, p.y - 1), INT_MAX },
  //  { Point(p.x - 1, p.y), INT_MAX },{ Point(p.x + 1, p.y), INT_MAX },
  //  { Point(p.x - 1, p.y + 1), INT_MAX },{ Point(p.x, p.y + 1), INT_MAX },{ Point(p.x + 1, p.y + 1), INT_MAX }
  //};

  PointsHueDistance adjacentPix;
  int dx = p.x - EndPoint.x;
  int dy = p.y - EndPoint.y;

  if (dx < 0)
  {
    if (dy < 0)
    {
      adjacentPix.push_back({ Point(p.x + 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y + 1), INT_MAX });
    }
    if (dy > 0)
    {
      adjacentPix.push_back({ Point(p.x - 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y + 1), INT_MAX });
    }
    if (dy == 0)
    {
      adjacentPix.push_back({ Point(p.x, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y + 1), INT_MAX });
    }
  }
  if (dx > 0)
  {
    if (dy < 0)
    {
      adjacentPix.push_back({ Point(p.x - 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y + 1), INT_MAX });
    }
    if (dy > 0)
    {
      adjacentPix.push_back({ Point(p.x - 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y + 1), INT_MAX });
    }
    if (dy == 0)
    {
      adjacentPix.push_back({ Point(p.x, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y + 1), INT_MAX });
    }
  }
  if (dx == 0)
  {
    if (dy < 0)
    {
      adjacentPix.push_back({ Point(p.x - 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y), INT_MAX });
    }
    if (dy > 0)
    {
      adjacentPix.push_back({ Point(p.x - 1, p.y), INT_MAX });
      adjacentPix.push_back({ Point(p.x - 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(p.x + 1, p.y), INT_MAX });
    }
  }

  // loop through adjacent pixels and update their hue distance
  auto _begin = adjacentPix.begin(), _end = adjacentPix.end();
  while (_begin != _end) {
    PointHueDistance& item = *_begin;
    ++_begin;
    if (item.p == last) continue;
    item.dHue = getHueDistance(Img, p, item.p);
  }

  // find point with minimal hue distance
  PointHueDistance ret = *min_element(adjacentPix.begin(), _end, cmpPointHueDistance);
  cout << ret.p << endl << ret.dHue << endl << endl;
  return ret.p;
}

//static void onMouse(int event, int x, int y, int, void* Img) {
//  // Proceed only if left mouse button was pressed
//  if (event != EVENT_LBUTTONDOWN) {
//    return;
//  }
//  Mat* bla = (Mat*)Img;
//  int weight;
//  Point blab = Point(x, y);
//  bla->at<Vec3b>(y, x) = Vec3b(0, 0, 255);
//
//  weight = getHueDistance(*bla, Point(x, y), Point(x, y + 1));
//
//  cout << Point(x, y) << endl
//    << Point(x, y + 1) << endl
//    << weight << endl;
//
//  imshow("Output", *bla);
//  //updateWindow("Output"); // add openGL to system
//}



int main(int argc, char** argv) {
  Mat ImgOrig;
  //Point start = {1, 1};

  // Check if image path is supplied as argument
  if (argc < 2) {
    cout << "Path must be applied as commandline argument." << endl;
    return -1;
  }

  // Read image and check if successfull
  ImgOrig = imread(argv[1]);
  if (ImgOrig.empty()) {
    cout << "Could not open or find the image." << endl;
    return -1;
  }

  // Convert given image to gray edge image
  //Mat ImgEdge = convertImgToEdge(ImgOrig);
  Mat ImgEdge = convertImgToGray(ImgOrig);

  // Create a window for display
  namedWindow("Output");
  // Listen to mouse events
  //setMouseCallback("Output", onMouse, &ImgEdge);
  // Display imge in window
  imshow("Output", ImgEdge);
  //imshow("Output", ImgOrig);

  Point temp = StartPoint;
  Point last = StartPoint;
  while (
    (temp.x - EndPoint.x) != 0
    || (temp.y - EndPoint.y) != 0
  ) {
    Point x = getNextPix(ImgEdge, temp, last);
    last = temp;
    temp = x;
  }
  

  // Wait for a keystroke in the window
  waitKey();
  return 0;
}