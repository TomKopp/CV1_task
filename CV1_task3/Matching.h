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
	cv::Mat _TmpTable;
	cv::Mat _ImgL;
	cv::Mat _ImgR;
	int _windowSize = 1;



	double calculateColorDistance(
		int heightl, int widthl, unsigned char *imgl, int il, int jl,
		int heightr, int widthr, unsigned char *imgr, int ir, int jr,
		int wsize
	) {
		// filter out points that are close to borders
	//if (il - wsize < 0) { return std::numeric_limits<double>::max(); }
	//if (ir - wsize < 0) { return std::numeric_limits<double>::max(); }
	//if (il + wsize > heightl - 1) { return std::numeric_limits<double>::max(); }
	//if (ir + wsize > heightr - 1) { return std::numeric_limits<double>::max(); }
	//if (jl - wsize < 0) { return std::numeric_limits<double>::max(); }
	//if (jr - wsize < 0) { return std::numeric_limits<double>::max(); }
	//if (jl + wsize > widthl - 1) { return std::numeric_limits<double>::max(); }
	//if (jr + wsize > widthr - 1) { return std::numeric_limits<double>::max(); }

	if (il - wsize < 0) { return INT_MAX; }
	if (ir - wsize < 0) { return INT_MAX; }
	if (il + wsize > heightl - 1) { return INT_MAX; }
	if (ir + wsize > heightr - 1) { return INT_MAX; }
	if (jl - wsize < 0) { return INT_MAX; }
	if (jr - wsize < 0) { return INT_MAX; }
	if (jl + wsize > widthl - 1) { return INT_MAX; }
	if (jr + wsize > widthr - 1) { return INT_MAX; }

		double q = 0.;
		for (int di = -wsize; di <= wsize; di++) {
			for (int dj = -wsize; dj <= wsize; dj++) {
				int indexl = ((il + di)*widthl + jl + dj) * 3;
				int indexr = ((ir + di)*widthr + jr + dj) * 3;
				double dr = std::abs(imgl[indexl] - imgr[indexr]);
				double dg = std::abs(imgl[indexl + 1] - imgr[indexr + 1]);
				double db = std::abs(imgl[indexl + 2] - imgr[indexr + 2]);
				q += dr + dg + db;
			}
		}
		return q;
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
		// "weighted frequency method" by Habr can work on asymetric matrices
		_CostTable = cv::Mat(pointsl.size(), pointsr.size(), CV_64F, 0.0);

		// fill _CostTable
		for (size_t r = 0; r < _PointsL.size(); r++) {
			for (size_t c = 0; c < _PointsR.size(); c++) {
				_CostTable.at<double>(r, c) = calculateColorDistance(
					_ImgL.rows, _ImgL.cols, _ImgL.ptr(0), (int)(_PointsL[r].y + 0.5), (int)(_PointsL[r].x + 0.5),
					_ImgR.rows, _ImgR.cols, _ImgR.ptr(0), (int)(_PointsR[c].y + 0.5), (int)(_PointsR[c].x + 0.5),
					_windowSize
				);
			}
		}

		_TmpTable = _CostTable.clone();
		double tableAvg = 0.0;
		// subtract row avg from _TmpTable
		double avg = 0.0;
		for (size_t r = 0; r < _CostTable.rows; r++) {
			avg = 0.0;
			for (size_t c = 0; c < _CostTable.cols; c++) {
				avg += _CostTable.at<double>(r, c);
			}
			avg /= _CostTable.rows;
			for (size_t c = 0; c < _CostTable.cols; c++) {
				_TmpTable.at<double>(r, c) -= avg;
			}
			tableAvg += avg;
		}

		// subtract column avg from _TmpTable
		for (size_t c = 0; c < _CostTable.cols; c++) {
			avg = 0.0;
			for (size_t r = 0; r < _CostTable.rows; r++) {
				avg += _CostTable.at<double>(r, c);
			}
			avg /= _CostTable.rows;
			for (size_t r = 0; r < _CostTable.rows; r++) {
				_TmpTable.at<double>(r, c) -= avg;
			}
			tableAvg += avg;
		}

		// add table average to elements
		tableAvg /= (_CostTable.rows + _CostTable.cols);
		for (size_t r = 0; r < _CostTable.rows; r++) {
			for (size_t c = 0; c < _CostTable.cols; c++) {
				_TmpTable.at<double>(r, c) += tableAvg;
			}
		}

		// find matrix minimum
		size_t maxNumberMatches = _PointsL.size() < _PointsR.size() ? _PointsL.size() : _PointsR.size();
		double minimum = 0.0;
		size_t count = 0;
		cv::Point minLoc(0,0);
		do {
			// find minimum in matrix
			cv::minMaxLoc(_TmpTable, &minimum, NULL, &minLoc, NULL);
			//if (minimum < std::numeric_limits<double>::max()) {
			if (minimum < INT_MAX) {
				// remove column and row of found minimum
				for (size_t r = 0; r < _TmpTable.rows; r++) {
					_TmpTable.at<double>(r, minLoc.x) = std::numeric_limits<double>::max();
				}
				for (size_t c = 0; c < _TmpTable.cols; c++) {
					_TmpTable.at<double>(minLoc.y, c) = std::numeric_limits<double>::max();
				}
				// save match
				MATCH match;
				match.value = minimum;
				match.xl = _PointsL[minLoc.y].x;
				match.yl = _PointsL[minLoc.y].y;
				match.xr = _PointsR[minLoc.x].x;
				match.yr = _PointsR[minLoc.x].y;
				_Matches.push_back(match);
			}
			count++;
		} while (minimum < std::numeric_limits<double>::max() && count < maxNumberMatches);

	}

	~Matching() {}

	std::vector<MATCH> getMatches() {
		return _Matches;
	}
};
