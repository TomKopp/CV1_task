#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

Mat ImgOrig, ImgGray, ImgRes;
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

static Mat convertImgToBGR(Mat& ImgOrig) {
  Mat Img;

  Img = ImgOrig.clone();
  cvtColor(Img, Img, COLOR_GRAY2BGR);

  return Img;
}

static int getHueDistance(Mat& Img, Point A, Point B) {
  uchar A_val, B_val, deltaHue;

  // ckeck if Points are within Mat, else the hue distance is huge
  if (
    (A.x < 0
    || A.y < 0
    || A.x > Img.cols -1
    || A.y > Img.rows -1)
    ||
    (B.x < 0
    || B.y < 0
    || B.x > Img.cols -1
    || B.y > Img.rows -1)
  ) {
    return INT_MAX;
  }

  // calculate hue distance
  A_val = Img.at<uchar>(A);
  B_val = Img.at<uchar>(B);
  deltaHue = A_val - B_val;

  // distances are always positive; mathematically correct would be sqrt(pow((double)(A_val - B_val), 2)) - but overkill
  if (deltaHue < 0) {
    deltaHue *= (-1);
  }

  return (int)deltaHue;
}

static Point getNextPix(Mat& Img, Vertices& List, Point P) {
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
    // if Point was already choosen before, don't update the hue distance
    if ( find(List.begin(), List.end(), item.p) != List.end() ) continue;
    item.dHue = getHueDistance(Img, P, item.p);
  }

  // find point with minimal hue distance
  PointHueDistance ret = *min_element(adjacentPix.begin(), _end, cmpPointHueDistance);
  return ret.p;
}

static void drawPathBGR(Mat Img, Vertices Path, uchar blue = 0, uchar green = 0, uchar red = 0) {
  // draw every Point from Path, colors need to be given else it's black
  auto _begin = Path.begin(), _end = Path.end();
  while (_begin != _end) {
    Point& item = *_begin;
    ++_begin;
    Img.at<Vec3b>(item) = Vec3b(blue, green, red);
  }
}

static void onMouse(int event, int x, int y, int, void* PointsList) {
  // proceed only if left mouse button was pressed
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
    //return;
  }

  // runs the path discovery - builds a list of Point's that define the path
  while (List->back() != EndPoint) {
    List->push_back( getNextPix(ImgGray, *List, List->back()) );
  }

  // print the path in the image
  drawPathBGR(ImgRes, *List, 0, 255);
  imshow("Result", ImgRes);
}



int main(int argc, char** argv) {
  Vertices PointsList = {};

  // check if image path is supplied as argument
  if (argc < 2) {
    cout << "Path must be applied as commandline argument." << endl;
    return -1;
  }

  // read image and check if successful
  ImgOrig = imread(argv[1]);
  if (ImgOrig.empty()) {
    cout << "Could not open or find the image." << endl;
    return -1;
  }

  // convert given image to gray edge image
  ImgGray = convertImgToGray(ImgOrig);
  ImgRes = convertImgToBGR(ImgGray);

  // create a window for display
  namedWindow("Output");
  namedWindow("Result");
  // listen to mouse events
  setMouseCallback("Output", onMouse, &PointsList);
  // display imge in window
  imshow("Output", ImgOrig);
  

  // wait for a keystroke in the window
  waitKey();
  return 0;
}