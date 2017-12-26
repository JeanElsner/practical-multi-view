#ifndef OPENCV_FAST_FEATURE_EXTRACTOR_H
#define OPENCV_FAST_FEATURE_EXTRACTOR_H
#include "BaseFeatureExtractor.h"
#include <opencv2/features2d.hpp>

class OpenCVFASTFeatureExtractor :
	public BaseFeatureExtractor
{
public:
	int threshold = 10;
	bool nonmax = true;

	virtual std::vector<Feature> extractFeatures(Frame& src, int max);
};
#endif
