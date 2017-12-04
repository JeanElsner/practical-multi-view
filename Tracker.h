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
	int init_frames = 5;
	int init_features = 80;
	int grid_size[2] = {255, 255};

	bool verbose = true;

	std::vector<Feature> features;
	std::vector<Frame> frames;

	BaseFeatureExtractor* extractor;
	BaseFeatureMatcher* matcher;

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
	std::vector<Frame> getGridROI(Frame& fr);

	/**
		Compute the standard deviation of a list of values

		@param val List of values
		@return The standard deviation
	*/
	double standardDeviation(std::vector<double> val);

	/**
		Divides the image into a grid, so as to distribute
		the extracted features across the entire image.

		@param I Input image
	*/
	void gridFeatureExtraction(
		Frame& I
	)
	{
		int gr = std::ceil((double)I.bw.rows / (double)grid_size[0]);
		int gc = std::ceil((double)I.bw.cols / (double)grid_size[1]);
		int fn = 0;
		std::vector<Feature> feats;

		for (int r = 0; r < I.bw.rows; r += grid_size[0])
		{
			for (int c = 0; c < I.bw.cols; c += grid_size[1])
			{
				cv::Rect rec(
					c, r, std::min((int)grid_size[1], I.bw.cols - c), 
					std::min((int)grid_size[0], I.bw.rows - r)
				);
				Frame roi = I.regionOfInterest(rec);

				int count = 0;
				std::vector<Feature> new_feats = extractor->extractFeatures(roi, fn);

				for (auto& f : new_feats)
				{
					f.column = c + f.column;
					f.row = r + f.row;

					if (count >= fn)
						break;

					if (!f.hasNeighbor(feats))
					{
						count++;
						feats.push_back(f);
					}
				}
			}
		}
	}

private:
	void countTrackedFeatures();
};
#endif
