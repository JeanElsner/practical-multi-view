#pragma once
#include <opencv2/core.hpp>
#include "BaseFeatureExtractor.h"

class ShiTomasiFeatureExtractor :
	public BaseFeatureExtractor
{
public:
	// Minimum quality of features, in order to be selected
	double quality = 0.3;

	ShiTomasiFeatureExtractor() {}

	virtual std::vector<Feature> extractFeatures(Frame& src, int max);

	cv::Mat computeShiTomasiResponse(Frame& src);
};
