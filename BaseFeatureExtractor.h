#ifndef BASE_FEATURE_EXTRACTOR_H
#define BASE_FEATURE_EXTRACTOR_H
#include <opencv2/core.hpp>
#include <vector>
#include "Feature.h"
#include "Frame.h"

// Abstract base class, used to extract features from images
class BaseFeatureExtractor
{
public:

	/**
		Extracts features from the given image (-section)

		@param src Image to extract features from
		@param max Maximum number of features to extract

		@return Vector of all the found features
	*/
	virtual std::vector<Feature> extractFeatures(Frame& src, int max) = 0;
};
#endif
