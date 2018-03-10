#ifndef KNN_FEATURE_MATCHER_H
#define KNN_FEATURE_MATCHER_H

#include "BaseFeatureMatcher.h"
#include "BaseFeatureExtractor.h"

class kNNFeatureMatcher : public BaseFeatureMatcher
{
private:
	int window = 15;
	int threshold = 2;
	BaseFeatureExtractor* extractor;
public:

	kNNFeatureMatcher(BaseFeatureExtractor* extractor) : extractor(extractor) {}

	virtual fmap matchFeatures(Frame& src, Frame& next);

	/**
		Finds a neighborhood around a feature

		@param f Center of the neighborhood
		@param feats List of features
		@param n Size of neighborhood
		@return A list of the nearest neighbors
	**/
	std::vector<Feature> getNearestNeighbors(
		Feature& f, const std::vector<Feature>& feats, int n = 7);

	/**
		Compare the pixel window around two featurs

		@param src The source image
		@param src_x The feature's x coordinate in the source image
		@param src_y The feature's y coordinate in the source image
		@param cmp Target image for comparison
		@param cmp_x The feature's x coordinate in the target image
		@param cmp_y The feature's y coordinate in the target image
		@return Average pixel error between the windows
	**/
	float compareFeatures(
		const cv::Mat& src, int src_x, int src_y, const cv::Mat& cmp, int cmp_x, int cmp_y);
};
#endif
