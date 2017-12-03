#ifndef BASE_FEATURE_MATCHER_H
#define BASE_FEATURE_MATCHER_H
#include "Frame.h"
#include "Feature.h"
#include <vector>

// Abstract base class, used to match features between frames
class BaseFeatureMatcher
{
public:

	/**
		Match a set of features from one frame and
		try to find them in another.

		@param src Source image with features to match
		@param next Image to match
		@param feats Set of features from the source image
		@param new_feats Output vector
	*/
	virtual void matchFeatures(
		Frame& src, 
		Frame& next, 
		std::vector<Feature>& feats, 
		std::vector<Feature>& new_feats
	) = 0;
};
#endif
