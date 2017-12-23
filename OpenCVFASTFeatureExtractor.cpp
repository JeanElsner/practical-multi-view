#include "OpenCVFASTFeatureExtractor.h"
#include "Tracker.h";

std::vector<Feature> OpenCVFASTFeatureExtractor::extractFeatures(Frame& src, int max)
{
	std::vector<Feature> feats;
	std::vector<cv::KeyPoint> kp;
	cv::FAST(src.bw, kp, threshold, nonmax);
	int i = 0;

	for (auto const& k : kp)
	{
		if (i >= max)
			break;
		i++;
		Feature f(k.pt);
		f.score = k.response;
		f.tracked = true;
		feats.push_back(f);
	}
	return feats;
}