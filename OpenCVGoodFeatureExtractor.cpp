#include "OpenCVGoodFeatureExtractor.h"
#include <opencv2/imgproc.hpp>

std::vector<Feature> OpenCVGoodFeatureExtractor::extractFeatures(Frame& src, int max)
{
	std::vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(src.bw, corners, max, quality, min_distance, cv::Mat(), 3, 3, false, 0.04);

	std::vector<Feature> feats;

	for (auto const& c : corners)
	{
		Feature f;
		f.row = c.y;
		f.column = c.x;
		f.detector = Feature::extractor::cv_good;
		f.tracked = true;
		feats.push_back(f);
	}
	return feats;
}
