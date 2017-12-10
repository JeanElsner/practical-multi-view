#include "OpenCVFASTFeatureExtractor.h"
#include "Tracker.h";

std::vector<Feature> OpenCVFASTFeatureExtractor::extractFeatures(Frame& src, int max)
{
	std::vector<Feature> feats;
	std::vector<cv::KeyPoint> kp;
	cv::FAST(src.bw, kp, threshold, nonmax);
	
	for (auto const& k : kp)
	{
		Feature f(k.pt);
		f.score = k.response;
		feats.push_back(f);
	}
	return feats;
}