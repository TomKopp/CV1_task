#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

Mat ImgOrig, ImgEdge;
Point StartPoint = {-1, -1};
Point EndPoint = {-1, -1};


struct PointHueDistance
{
  Point p;
  int dHue;
};

typedef vector<PointHueDistance> PointsHueDistance;
typedef vector<Point> Vertices;


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

static Point getNextPix(Mat& Img, Point P, Point Last) {
  // all adjacent points with initial hue distance
  //PointsHueDistance adjacentPix = {
  //  { Point(p.x - 1, p.y - 1), INT_MAX },{ Point(p.x, p.y - 1), INT_MAX },{ Point(p.x + 1, p.y - 1), INT_MAX },
  //  { Point(p.x - 1, p.y), INT_MAX },{ Point(p.x + 1, p.y), INT_MAX },
  //  { Point(p.x - 1, p.y + 1), INT_MAX },{ Point(p.x, p.y + 1), INT_MAX },{ Point(p.x + 1, p.y + 1), INT_MAX }
  //};

  PointsHueDistance adjacentPix;
  int dx = P.x - EndPoint.x;
  int dy = P.y - EndPoint.y;

  // all direction aware adjacent points with initial hue distance
  if (dx < 0)
  {
    if (dy < 0)
    {
      adjacentPix.push_back({ Point(P.x + 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y + 1), INT_MAX });
    }
    if (dy > 0)
    {
      adjacentPix.push_back({ Point(P.x - 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y + 1), INT_MAX });
    }
    if (dy == 0)
    {
      adjacentPix.push_back({ Point(P.x, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y + 1), INT_MAX });
    }
  }
  if (dx > 0)
  {
    if (dy < 0)
    {
      adjacentPix.push_back({ Point(P.x - 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y + 1), INT_MAX });
    }
    if (dy > 0)
    {
      adjacentPix.push_back({ Point(P.x - 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y + 1), INT_MAX });
    }
    if (dy == 0)
    {
      adjacentPix.push_back({ Point(P.x, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y + 1), INT_MAX });
    }
  }
  if (dx == 0)
  {
    if (dy < 0)
    {
      adjacentPix.push_back({ Point(P.x - 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y + 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y), INT_MAX });
    }
    if (dy > 0)
    {
      adjacentPix.push_back({ Point(P.x - 1, P.y), INT_MAX });
      adjacentPix.push_back({ Point(P.x - 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y - 1), INT_MAX });
      adjacentPix.push_back({ Point(P.x + 1, P.y), INT_MAX });
    }
  }

  // loop through adjacent pixels and update their hue distance
  auto _begin = adjacentPix.begin(), _end = adjacentPix.end();
  while (_begin != _end) {
    PointHueDistance& item = *_begin;
    ++_begin;
    if (item.p == Last) continue;
    item.dHue = getHueDistance(Img, P, item.p);
  }

  // find point with minimal hue distance
  PointHueDistance ret = *min_element(adjacentPix.begin(), _end, cmpPointHueDistance);
  cout << ret.p << endl << ret.dHue << endl << endl;
  return ret.p;
}

static void onMouse(int event, int x, int y, int, void* PointsList) {
  // Proceed only if left mouse button was pressed
  if (event != EVENT_LBUTTONDOWN) {
    return;
  }
  Vertices* List = (Vertices*)PointsList;

  // first click sets StartPoint
  if (StartPoint == Point(-1, -1)) {
    StartPoint = Point(x, y);
    List->push_back(Point(x, y));
    List->push_back(Point(x, y)); // StartPoint needs to be pushed TWO times
    return;
  }
  // second click sets EndPoint and...
  if (EndPoint == Point(-1, -1)) {
    EndPoint = Point(x, y);
    return;
  }
  // runs the path discovery - builds a list of Point's that define the path
  while (List->back() != EndPoint) {
    List->push_back(getNextPix(ImgEdge, List->back(), *(List->end() -2) ));
  }

  // Print the path in the image

  /*Mat* bla = (Mat*)Img;
  int weight;
  Point blab = Point(x, y);
  bla->at<Vec3b>(y, x) = Vec3b(0, 0, 255);

  weight = getHueDistance(*bla, Point(x, y), Point(x, y + 1));

  cout << Point(x, y) << endl
    << Point(x, y + 1) << endl
    << weight << endl;

  imshow("Output", *bla);*/
  //updateWindow("Output"); // add openGL to system
}



int main(int argc, char** argv) {
  Vertices PointsList = {};
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
  ImgEdge = convertImgToGray(ImgOrig);

  // Create a window for display
  namedWindow("Output");
  // Listen to mouse events
  setMouseCallback("Output", onMouse, &PointsList);
  // Display imge in window
  imshow("Output", ImgEdge);
  //imshow("Output", ImgOrig);
  

  // Wait for a keystroke in the window
  waitKey();
  return 0;
}