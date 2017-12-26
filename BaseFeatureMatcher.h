#ifndef BASE_FEATURE_MATCHER_H
#define BASE_FEATURE_MATCHER_H
#include "Frame.h"
#include "Feature.h"
#include <vector>
#include <unordered_map>

// Abstract base class, used to match features between frames
class BaseFeatureMatcher
{
public:

	typedef std::unordered_map < Feature, Feature, Feature::Hasher> fmap;

	/**
		Match a set of features from one frame and
		try to find them in another.

		@param src Source image with features to match
		@param next Image to match
	*/
	virtual fmap matchFeatures(Frame& src, Frame& next) = 0;
};
#endif
