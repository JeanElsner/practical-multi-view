#include "ShiTomasiFeatureExtractor.h"

std::vector<Feature> ShiTomasiFeatureExtractor::extractFeatures(Frame & src, int max)
{
	cv::Mat R = computeShiTomasiResponse(src);
	std::vector<Feature> feats;

	cv::Mat thresh(src.bw.size(), CV_64FC1);
	double r_min, r_max;
	minMaxLoc(R, &r_min, &r_max);
	threshold(R, thresh, r_max*0.4, 255, CV_THRESH_BINARY);

	for (int j = 0; j < src.bw.rows; j++)
	{
		double* p_thresh = thresh.ptr<double>(j);
		double* p_R = R.ptr<double>(j);

		for (int i = 0; i < src.bw.cols; i++)
		{
			if (p_thresh[i] == 255)
			{
				Feature f;
				f.row = j;
				f.column = i;
				f.detector = Feature::extractor::shi_tomasi;
				feats.push_back(f);
			}
		}
	}
	return feats;
}

cv::Mat ShiTomasiFeatureExtractor::computeShiTomasiResponse(Frame& src)
{
	cv::Mat dst = cv::Mat::zeros(src.bw.size(), CV_64FC1);

	for (int r = 0; r < src.bw.rows; r++)
	{
		const double* p_H = src.getHarrisMatrix().ptr<double>(r);
		double* p_dst = dst.ptr<double>(r);

		for (int c = 0; c < (src.getHarrisMatrix().cols - 1) * 3; c += 3)
		{
			double Ixx = p_H[c + 0];
			double Iyy = p_H[c + 1];
			double Ixy = p_H[c + 2];

			double B = -Ixx - Iyy;
			double C = Ixx*Iyy - pow(Ixy, 2);

			double lambda_1 = (-B + sqrt(pow(B, 2) - 4 * C)) / 2;
			double lambda_2 = (-B - sqrt(pow(B, 2) - 4 * C)) / 2;

			//p_dst[c / 3] = lambda_1*lambda_2 - 0.04f*pow(lambda_1 +lambda_2, 2);
			p_dst[c / 3] = std::min(lambda_1, lambda_2);
		}
	}
	return dst;
}
