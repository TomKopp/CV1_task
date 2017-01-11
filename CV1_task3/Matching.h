#pragma once

#include <vector>
#include <opencv2/core/core.hpp>

#include "ps.h"

class Matching
{
private:
	std::vector<MATCH> _Matches;
	std::vector<KEYPOINT> _PointsL;
	std::vector<KEYPOINT> _PointsR;
	cv::Mat _CostTable;
	cv::Mat _ImgL;
	cv::Mat _ImgR;
	int _windowSize = 1;



	double calculateColorDistance(
		int heightl, int widthl, unsigned char *imgl, int il, int jl,
		int heightr, int widthr, unsigned char *imgr, int ir, int jr,
		int wsize
	) {
		// filter out points that are close to borders
		if (il - wsize < 0) { return 0.0; }
		if (ir - wsize < 0) { return 0.0; }
		if (il + wsize > heightl - 1) { return 0.0; }
		if (ir + wsize > heightr - 1) { return 0.0; }
		if (jl - wsize < 0) { return 0.0; }
		if (jr - wsize < 0) { return 0.0; }
		if (jl + wsize > widthl - 1) { return 0.0; }
		if (jr + wsize > widthr - 1) { return 0.0; }

		double q = 0.;
		for (int di = -wsize; di <= wsize; di++) {
			for (int dj = -wsize; dj <= wsize; dj++) {
				int indexl = ((il + di)*widthl + jl + dj) * 3;
				int indexr = ((ir + di)*widthr + jr + dj) * 3;
				double dr = (double)imgl[indexl] - (double)imgr[indexr];
				double dg = (double)imgl[indexl + 1] - (double)imgr[indexr + 1];
				double db = (double)imgl[indexl + 2] - (double)imgr[indexr + 2];
				q += dr*dr + dg*dg + db*db;
			}
		}
		return q;
	}

	void fillCostTable()
	{
		double color_distance = 0.0;
		for (size_t l = 0; l < _PointsL.size(); l++) {
			for (size_t r = 0; r < _PointsR.size(); r++) {
				_CostTable.at<double>(l, r) = calculateColorDistance(
					_ImgL.rows, _ImgL.cols, _ImgL.ptr(0), (int)(_PointsL[l].y + 0.5), (int)(_PointsL[l].x + 0.5),
					_ImgR.rows, _ImgR.cols, _ImgR.ptr(0), (int)(_PointsR[r].y + 0.5), (int)(_PointsR[r].x + 0.5),
					_windowSize
				);
			}
		}
	};

	void subtractRowMinima()
	{
		double minimum = _CostTable.at<double>(0, 0);
		for (size_t r = 0; r < _CostTable.rows; r++) {
			for (size_t c = 0; c < _CostTable.cols; c++) {
				if (minimum > _CostTable.at<double>(r, c)) {
					minimum = _CostTable.at<double>(r, c);
				}
			}
			for (size_t c = 0; c < _CostTable.cols; c++) {
				_CostTable.at<double>(r, c) -= minimum;
			}
		}
	}

	void subtractColumnMinima()
	{
		double minimum = _CostTable.at<double>(0, 0);
		for (size_t c = 0; c < _CostTable.cols; c++) {
			for (size_t r = 0; r < _CostTable.cols; r++) {
				if (minimum > _CostTable.at<double>(r, c)) {
					minimum = _CostTable.at<double>(r, c);
				}
			}
			for (size_t r = 0; r < _CostTable.cols; r++) {
				_CostTable.at<double>(r, c) -= minimum;
			}
		}
	}

public:
	Matching(
		const cv::Mat & ImgL,
		const cv::Mat & ImgR,
		std::vector<KEYPOINT> & pointsl,
		std::vector<KEYPOINT> & pointsr,
		int wsize
	) :
		_ImgL(ImgL),
		_ImgR(ImgR),
		_PointsL(pointsl),
		_PointsR(pointsr),
		_windowSize(wsize)
	{
		size_t size = pointsl.size() > pointsr.size() ? pointsl.size() : pointsr.size();
		_CostTable = cv::Mat(cv::Size(size, size), CV_64F, 0.0);
		fillCostTable();
	}

	~Matching() {}
};
