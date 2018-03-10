#ifndef OPENCV_GOOD_FEATURE_EXTRACTOR_H
#define OPENCV_GOOD_FEATURE_EXTRACTOR_H
#include "BaseFeatureExtractor.h"

class OpenCVGoodFeatureExtractor: public BaseFeatureExtractor
{
public:
	// Minimum quality of features, in order to be selected
	double quality = 0.01;
	// Minimum distance between features, in order to be considered separate
	double min_distance = 5;

	OpenCVGoodFeatureExtractor() { }

	OpenCVGoodFeatureExtractor(double quality, double min_distance): 
		quality(quality), 
		min_distance(min_distance) { }

	std::vector<Feature> extractFeatures(Frame& src, int max);
};
#endif
