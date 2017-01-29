#include "ObservationModel.h"

void calculateHist(cv::Mat& img, cv::Mat& hist) {
	int histSize[3];
	float hranges[2];
	const float* ranges[3];
	int channels[3];
	// Prepare arguments for a color histogram
	histSize[0] = 5; histSize[1] = histSize[2] = 25;
	hranges[0] = 0.0; // BRG range
	hranges[1] = 255.0;
	ranges[0] = hranges; // all channels have the same range
	ranges[1] = hranges;
	ranges[2] = hranges;
	channels[0] = 0; // the three channels
	channels[1] = 1;
	channels[2] = 2;

	cv::calcHist(&img,
		1, // histogram of 1 image only
		channels, // the channel used
		cv::Mat(), // no mask is used
		hist, // the resulting histogram
		3, // it is a 3D histogram
		histSize, // number of bins
		ranges // pixel value range
	);
}


ObservationModel::ObservationModel(cv::Mat & img, double lambda_)
	:_lambda(lambda_)
{
	// IMPLEMENT - DONE
	// claculate the histogram of the given image (use calcHist function above)
	calculateHist(img, _histogram);
	// normalize the histogram - not realy nessesary
	//cv::normalize(_histogram, _histogram);
	// store the histogram and lambda
}

double ObservationModel::likelihood(cv::Mat & img, Particle & p) {
	// IMPLEMENT - DONE
	// caluclate a histogramm for the window defined by the particle
	cv::Mat histWindow;
	calculateHist(p.getSubImg(img), histWindow);

	// normalize it - not realy nessesary
	//cv::normalize(histWindow, histWindow);

	// compare it to the stored histogram using the cv::compareHist function (use the Bahttacharyya distance)
	double histDistance = cv::compareHist(_histogram, histWindow, CV_COMP_BHATTACHARYYA);
	// return the likelihood exp(-lambda * histogram_distance)

	return std::exp((-_lambda) * histDistance);
}

