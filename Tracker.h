#ifndef TRACKER_H
#define TRACKER_H
#include "Feature.h";
#include "Frame.h"
#include "BaseFeatureExtractor.h"
#include "BaseFeatureMatcher.h"
#include <vector>

class Tracker
{
public:
	int tracked_features = 0;
	int min_tracked_features = 50;

	std::vector<Feature> features;
	std::vector<Frame> frames;

	BaseFeatureExtractor* extractor;
	BaseFeatureMatcher* matcher;

	/**
		Adds a frame to the tracker

		@param frame Frame to add
	*/
	void addFrame(const Frame& frame) { frames.push_back(frame); }
};
#endif
