#pragma once
#include <opencv2/core.hpp>
#include "BaseFeatureExtractor.h"

class ShiTomasiFeatureExtractor :
	public BaseFeatureExtractor
{
public:
	// Minimum quality of features, in order to be selected
	double quality = 0.4;

	ShiTomasiFeatureExtractor() {}

	virtual std::vector<Feature> extractFeatures(Frame& src, int max);

	/**
		Compute the Shi-Tomasi response matrix, assigning
		each pixel a score based on the probability of it being
		a corner.

		@param src The image to analyse
		@return Image with an absolute score for each pixel
	*/
	cv::Mat computeShiTomasiResponse(Frame& src);

};
