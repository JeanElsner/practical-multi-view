#include "Frame.h"

Frame::Frame(cv::Mat & orig) : orig(orig)
{
	init();
}

Frame::Frame(const std::string& file_name)
{
	orig = cv::imread(file_name, cv::IMREAD_COLOR);
	init();
}

void Frame::init()
{
	bw = cv::Mat(orig.size(), CV_32FC1);
	cv::cvtColor(orig, bw, cv::COLOR_BGR2GRAY);
	harris = cv::Mat(orig.size(), CV_32FC3);
}

cv::Mat& Frame::getSpatialGradientX()
{
	if (!computed_gradient)
		computeSpatialGradient();
	return grad_x;
}

cv::Mat& Frame::getSpatialGradientY()
{
	if (!computed_gradient)
		computeSpatialGradient();
	return grad_y;
}

void Frame::computeSpatialGradient()
{
	grad_x = cv::Mat::zeros(bw.size(), CV_64FC1);
	grad_y = cv::Mat::zeros(bw.size(), CV_64FC1);

	for (int r = 1; r < bw.rows - 1; r++)
	{
		const schar* prev = bw.ptr<schar>(r - 1);
		const schar* curr = bw.ptr<schar>(r);
		const schar* next = bw.ptr<schar>(r + 1);

		double* p_Ix = grad_x.ptr<double>(r);
		double* p_Iy = grad_y.ptr<double>(r);

		for (int c = 1; c < bw.cols - 1; c++)
		{
			//p_Ix[c] = 1. * curr[c - 1] - 2. * curr[c] + 1. * curr[c + 1];
			//p_Iy[c] = 1. * prev[c] - 2. * curr[c] + 1. * next[c];

			p_Ix[c] = 1. / 2. * curr[c + 1] - 1. / 2. * curr[c - 1];
			p_Iy[c] = 1. / 2.*next[c] - 1. / 2.*prev[c];
			/*p_Ix[c] = 1.*prev[c - 1] - 1.*prev[c + 1] + 2. * curr[c - 1]
			- 2. * curr[c + 1] + 1.*next[c - 1] - 1.*next[c + 1];
			p_Iy[c] = 1.*prev[c - 1] + 2.*prev[c] +  1.*prev[c  + 1]
			- 1.*next[c - 1] - 2.*next[c] - 1.*next[c  + 1];*/
		}
	}
	computed_gradient = true;
}

cv::Mat& Frame::getHarrisMatrix()
{
	if (!computed_harris)
		computeHarrisMatrix();
	return harris;
}

void Frame::computeHarrisMatrix()
{
	cv::Mat* Ix = &getSpatialGradientX();
	cv::Mat* Iy = &getSpatialGradientY();

	cv::Mat Ixx = Ix->mul(*Ix);
	cv::Mat Iyy = Iy->mul(*Iy);
	cv::Mat Ixy = Ix->mul(*Iy);

	std::vector<cv::Mat> channels(3);
	cv::split(harris, channels);
	channels[0] = Ixx;
	channels[1] = Iyy;
	channels[2] = Ixy;
	cv::merge(channels, harris);

	cv::blur(harris, harris, cv::Size(3, 3));
}