#ifndef TRACKER_H
#define TRACKER_H
#include "Feature.h";
#include "Frame.h"
#include "BaseFeatureExtractor.h"
#include "BaseFeatureMatcher.h"
#include <vector>

class Tracker
{
private:
	bool init = false;

public:
	int tracked_features = 0;
	int min_tracked_features = 50;
	int tracked_features_tol = 5;
	int init_frames = 5;
	int init_features = 80;
	int grid_size[2] = {255, 255};

	bool verbose = true;

	std::vector<Feature> features;
	std::vector<Frame> frames;

	BaseFeatureExtractor* extractor;
	BaseFeatureMatcher* matcher;

	class GridSection
	{
	public:
		// Coordinates within the grid
		int x, y;

		Frame frame;

		std::vector<Feature> features;

		GridSection(Frame frame, int x, int y): frame(frame), x(x), y(y) { }
	};

	/**
		Standard constructor, assigning both a feature extractor
		as well as a matcher.

		@param extractor The feature extractor
		@param matcher The feature matcher
	*/
	Tracker(BaseFeatureExtractor* extractor, BaseFeatureMatcher* matcher) :
		extractor(extractor),
		matcher(matcher)
	{ }

	/**
		Adds a frame to the tracker extracting new
		features as needed.

		@param frame Frame to add
	*/
	void addFrame(Frame& frame);

	/**
		Initialises the tracker. That is, the frame with the
		best selection of features is chosen as a starting point
		for the tracker.
	*/
	void initialise();

	double computeFeatureCost(Frame& frame);

	/**
		Divides the frame into a grid of regions of interest
		according to this object's setting.

		@param fr The source frame to be divided
		@return A vector containing the grid
	*/
	std::vector<GridSection> getGridROI(Frame& fr);

	/**
		Compute the standard deviation of a list of values

		@param val List of values
		@return The standard deviation
	*/
	double standardDeviation(std::vector<double> val);

private:
	void countTrackedFeatures();
};
#endif
