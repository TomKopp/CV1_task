#pragma once

#include <array>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/// <summary>
/// A Combined Corner And Edge Detector - Harris & Stephens
/// </summary>
/// <remarks>
/// http://www.bmva.org/bmvc/1988/avc-88-023.pdf
/// X = I * (-1, 0, 1)
/// Y = I * (-1, 0, 1)T
/// E(x,y) = A(x^2) + Cxy + B(y^2)
/// A = X^2 * w
/// B = Y^2 * w
/// C = (XY) * w
/// w_u,v = exp -(u^2 + v^2) / 2 * sigma^2 -> Gaussian kernel
/// E(x,y) = (x, y) M (x, y)T
/// M = |A C|  // Structure tensor - second moment matrix - autocorrelation matrix
///     |C B|
/// alpha, beta = eigenvalues of M
/// Tr(M) = alpha + beta = A + B // Trace
/// Det(M) = alpha*beta = AB - C^2 // Determinant
/// Corner response:
/// R = Det - k * Tr^2
/// R is positive in the corner region, negative in the edge region and small in the flat region
/// The flat region is specified by Tr falling below some selected threshold
/// Corner region pixel is an 8-way local maximum
/// Edge region pixel if R negative and local minimum
/// </remarks>
class HarrisDetector
{
private:
	cv::Mat _ImgOrig;
	cv::Mat _Response;
	std::array<cv::Mat, 2> _Derivatives;

public:
	/// <summary>
	///
	/// </summary>
	struct KEYPOINT {
		double x;
		double y;
		double value;
	};

	typedef std::vector<HarrisDetector::KEYPOINT> KEYPOINTS;

private:
	/// <summary>
	/// Convolves the image with the kernel.
	/// </summary>
	/// <param name="Img">The img.</param>
	/// <param name="Kernel">The kernel.</param>
	/// <returns>cv::Mat</returns>
	cv::Mat _convolveKernel(const cv::Mat & Img, const cv::Mat & Kernel)
	{
		cv::Mat Ret(Img.size(), CV_32F, cv::Scalar(0.0));

		cv::filter2D(Img, Ret, CV_32F, Kernel);

		return Ret;
	}

	/// <summary>
	/// Convolves the image with a gaussian kernel.
	/// </summary>
	/// <remarks>
	/// Kernel from here: http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
	/// for sigma = 1
	/// </remarks>
	/// <param name="Img">The img.</param>
	/// <returns>cv::Mat</returns>
	cv::Mat _convolveGaussian(const cv::Mat & Img)
	{
		cv::Mat GaussianKernel = (cv::Mat_<float>(5, 5) <<
			1, 4, 7, 4, 1,
			4, 16, 26, 16, 4,
			7, 16, 41, 16, 7,
			4, 16, 26, 16, 4,
			1, 4, 7, 4, 1);
		GaussianKernel = GaussianKernel / 273.0;

		return _convolveKernel(Img, GaussianKernel);
	}

	/// <summary>
	/// Computes the derivatives.
	/// </summary>
	/// <param name="Img">The img.</param>
	/// <returns>std::array</returns>
	std::array<cv::Mat, 2> _computeDerivatives(const cv::Mat & Img)
	{
		std::array<cv::Mat, 2> Ret;

		Ret[0] = _convolveKernel(Img, (cv::Mat_<float>(1, 3) << -1, 0, 1)); // X = I * (-1, 0, 1)
		Ret[1] = _convolveKernel(Img, (cv::Mat_<float>(3, 1) << -1, 0, 1)); // Y = I * (-1, 0, 1)T

		return Ret;
	}

	/// <summary>
	/// Computes the Harris response for each element.
	/// All Structure tensor elements must have the same size.
	/// </summary>
	/// <param name="StructureTensor">The structure tensor.</param>
	/// <returns>cv::Mat</returns>
	cv::Mat _computeResponse(const std::array<cv::Mat, 3> & StructureTensor)
	{
		CV_Assert(
			StructureTensor[0].size() == StructureTensor[1].size()
			&& StructureTensor[0].size() == StructureTensor[2].size()
		);

		cv::Mat
			Ret(StructureTensor[0].size(), CV_32F, cv::Scalar(0.0)),
			A = StructureTensor[0],
			B = StructureTensor[1],
			C = StructureTensor[2];
		float
			k = 0.04f, // empirical constant: k = 0.04 - 0.06
			det, // Det = AB - C^2
			tr; // Tr = A + B

		for (int r = 0; r < Ret.rows; r++) {
			for (int c = 0; c < Ret.cols; c++) {
				det = A.at<float>(r, c) * B.at<float>(r, c) - C.at<float>(r, c) * C.at<float>(r, c); // Det = AB - C^2
				tr = A.at<float>(r, c) + B.at<float>(r, c); // Tr = A + B
				Ret.at<float>(r, c) = det - k * tr  * tr; // R = Det - k * Tr^2
			}
		}

		return Ret;
	}

	/// <summary>
	/// Performs the non-maxima suppression on a 3x3 neighboorhood.
	/// </summary>
	/// <remarks>
	/// https://www.academia.edu/5524439/Non-maximum_Suppression_Using_fewer_than_Two_Comparisons_per_Pixel
	/// </remarks>
	/// <param name="Response">The response.</param>
	/// <returns>cv::Mat</returns>
	cv::Mat _nonMaximaSuppression(const cv::Mat & Response)
	{
		cv::Mat Ret(Response.size(), CV_32F, cv::Scalar(0.0)); // == Mask
		int
			c, /// <value>column index</value>
			r, /// <value>row index</value>
			h = Response.rows - 1,
			w = Response.cols - 1,
			cur = 0,
			next = 1;

		bool(*skip)[2] = new bool[Response.cols][2]; // skanline mask
		for (int i = 0; i < Response.cols; ++i) { // initialize mask
			skip[i][0] = false;
			skip[i][1] = false;
		}

		for (r = 1; r < h - 1; ++r) {
			c = 1; // set c (column index) every start of the loop to two

			while (c < (w - 1)) {
				if (skip[c][cur]) { // skip current pixel
					++c;
					continue;
				}

				/* Scanline in 1D */
				if (Response.at<float>(r, c) <= Response.at<float>(r, c + 1)) {
					++c;
					while (c < w && Response.at<float>(r, c) <= Response.at<float>(r, c + 1)) { // compare pixels right neighbor with its right neighbor
						++c;
					}
					if (c == w) {
						break;
					}
				}
				else {
					if (Response.at<float>(r, c) <= Response.at<float>(r, c - 1)) {
						++c;
						continue;
					}
				}
				skip[c + 1][cur] = true;
				/********/

				// compare to 3 future then 3 past neighbors
				if (Response.at<float>(r, c) <= Response.at<float>(r + 1, c - 1)) { ++c; continue; }
				skip[c - 1][next] = true;

				if (Response.at<float>(r, c) <= Response.at<float>(r + 1, c)) { ++c; continue; }
				skip[c][next] = true;

				if (Response.at<float>(r, c) <= Response.at<float>(r + 1, c + 1)) { ++c; continue; }
				skip[c + 1][next] = true;

				if (Response.at<float>(r, c) <= Response.at<float>(r - 1, c - 1)) { ++c; continue; }
				if (Response.at<float>(r, c) <= Response.at<float>(r - 1, c)) { ++c; continue; }
				if (Response.at<float>(r, c) <= Response.at<float>(r - 1, c + 1)) { ++c; continue; }

				Ret.at<float>(r, c) = Response.at<float>(r, c);
				++c;
			}

			// swap skip mask indices
			std::swap(cur, next);
			for (int i = 0; i < Response.cols; ++i) { // reset next scanline mask
				skip[i][next] = false;
			}
		}

		delete[] skip;
		return Ret;
	}

	/// <summary>
	/// Performs the non-maxima suppression on a given neighboorhood size.
	/// Not implemented yet!
	/// </summary>
	/// <remarks>
	/// https://www.academia.edu/5524439/Non-maximum_Suppression_Using_fewer_than_Two_Comparisons_per_Pixel
	/// </remarks>
	/// <param name="Response">The response.</param>
	/// <param name="Neighborhood">The neighborhood defined as (2n + 1)×(2n + 1). Give n</param>
	/// <returns>cv::Mat</returns>
	cv::Mat _nonMaximaSuppression(const cv::Mat & Response, uchar Neighborhood)
	{
		return cv::Mat();
	}

public:
	/// <summary>
	/// Initializes a new instance of the <see cref="HarrisDetector"/> class.
	/// Calculates the Harris corner response.
	/// </summary>
	/// <param name="Img">The img.</param>
	HarrisDetector(const cv::Mat & Img)
		: _ImgOrig(Img.clone())
	{
		cv::Mat WorkingCopy;
		std::array<cv::Mat, 3> StructureTensor;
		_ImgOrig.convertTo(WorkingCopy, CV_32F);
		cv::cvtColor(_ImgOrig, WorkingCopy, cv::COLOR_BGR2GRAY);

		_Derivatives = _computeDerivatives(WorkingCopy);

		StructureTensor[0] = _convolveGaussian(_Derivatives[0].mul(_Derivatives[0])); // A = X^2 * w
		StructureTensor[1] = _convolveGaussian(_Derivatives[1].mul(_Derivatives[1])); // B = Y^2 * w
		StructureTensor[2] = _convolveGaussian(_Derivatives[0].mul(_Derivatives[1])); // C = (XY) * w

		_Response = _computeResponse(StructureTensor);
	}

	/// <summary>
	/// Finalizes an instance of the <see cref="HarrisDetector"/> class.
	/// </summary>
	~HarrisDetector() {}

	/// <summary>
	/// Gets the Harris response.
	/// </summary>
	/// <returns>cv::Mat</returns>
	cv::Mat getResponse()
	{
		cv::Mat Ret = _Response.clone();
		return Ret;
	}

	/// <summary>
	/// Gets the derivatives.
	/// </summary>
	/// <param name="raw">if set to <c>true</c> [raw].</param>
	/// <returns>std::array</returns>
	std::array<cv::Mat, 2> getDerivatives(bool raw = false)
	{
		std::array<cv::Mat, 2> Ret = {
			_Derivatives[0].clone(),
			_Derivatives[1].clone()
		};

		if (!raw) {
			Ret[0].convertTo(Ret[0], _ImgOrig.type());
			Ret[1].convertTo(Ret[1], _ImgOrig.type());
		}

		return Ret;
	}

	/// <summary>
	/// Filters the img by responses.
	/// If response value greater/lower/... than a threshold, determinded by the compare function,
	/// the value of the original Image is keept else it's blacked out.
	/// The compare function gets the float value of the Harris response and has to return a boolean.
	/// </summary>
	/// <param name="cmpFnc">The compare function.</param>
	/// <returns>cv::Mat</returns>
	template<typename Functor> inline
		cv::Mat filterImgByResponse(const Functor& cmpFnc)
	{
		cv::Mat Ret = _ImgOrig.clone();
		Ret.convertTo(Ret, CV_32F);

		for (int r = 0; r < Ret.rows; r++) {
			for (int c = 0; c < Ret.cols; c++) {
				if (!cmpFnc(_Response.at<float>(r, c))) {
					Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 0);
				}
			}
		}

		Ret.convertTo(Ret, _ImgOrig.type());
		return Ret;
	}

	/// <summary>
	/// Filters the corners of the Response after non-maxima suppression.
	/// </summary>
	/// <param name="cmpFnc">The compare function.</param>
	/// <returns>cv::Mat</returns>
	template<typename Functor> inline
		cv::Mat filterCorners(const Functor& cmpFnc)
	{
		cv::Mat Ret(_Response.size(), CV_32FC3, cv::Scalar::all(0.0));
		cv::Mat NMS = _nonMaximaSuppression(_Response);

		for (int r = 0; r < Ret.rows; r++) {
			for (int c = 0; c < Ret.cols; c++) {
				if (cmpFnc(NMS.at<float>(r, c))) {
					Ret.at<cv::Vec3f>(r, c) = cv::Vec3f(0, 0, 255);
				}
			}
		}

		Ret.convertTo(Ret, _ImgOrig.type());
		return Ret;
	}

	/// <summary>
	/// Filters the keypoints.
	/// </summary>
	/// <param name="cmpFunc">The compare function.</param>
	/// <returns></returns>
	template<typename Functor> inline
		KEYPOINTS filterKeyPoints(const Functor& cmpFnc)
	{
		KEYPOINTS Ret;
		cv::Mat NMS = _nonMaximaSuppression(_Response);

		for (int r = 0; r < NMS.rows; r++) {
			for (int c = 0; c < NMS.cols; c++) {
				if (cmpFnc(NMS.at<float>(r, c))) {
					KEYPOINT blub;
					blub.x = c;
					blub.y = r;
					blub.value = NMS.at<float>(r, c);
					Ret.push_back(blub);
				}
			}
		}

		return Ret;
	}
};
