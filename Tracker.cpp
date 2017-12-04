#include "Tracker.h"
#include <iostream>
#include <opencv2/highgui.hpp>

void Tracker::addFrame(Frame& frame)
{
	if (frames.size() >= init_frames && !init)
	{
		initialise();
	}
	frame.frame = frames.size();
	frames.push_back(frame);

	if (init)
	{
		Frame& src = frames[frames.size() - 2];
		Frame& next = frames[frames.size() - 1];
		std::vector<Feature> new_feats;

		matcher->matchFeatures(src, next, features, new_feats);

		for (int i = 0; i < new_feats.size(); i++)
		{
			if (new_feats[i].tracked)
			{
				circle(frame.orig, new_feats[i].point(), 5, cv::Scalar(0, 255, 255));
				line(frame.orig, features[i].point(), new_feats[i].point(), cv::Scalar(0, 255, 255));
			}
		}
		features = new_feats;

		cv::imshow("test", frame.orig);
		cv::waitKey(20);
	}
	countTrackedFeatures();

	if (init && tracked_features + tracked_features_tol < min_tracked_features)
	{
		int n = min_tracked_features - tracked_features;

		if (verbose)
			std::cout << "Trying to find " << n << " new features" << 
			" in frame #" << (frames.size() - 1) << std::endl;

		std::vector<Feature> new_feats = extractor->extractFeatures(frames[frames.size() - 1], n);

		for (auto& f : new_feats)
		{
			if (!f.hasNeighbor(features))
				features.push_back(f);
		}
	}
}

void Tracker::initialise()
{
	if (verbose)
		std::cout << "Initialising..." << std::endl;

	features.clear();
	Frame* best = &frames[0];
	double cost = HUGE_VAL;

	for (auto& fr : frames)
	{
		std::vector<GridSection> roi = getGridROI(fr);
		double n = init_features / roi.size();
		double std_n = 0;
		std::vector<double> n_i;
		double std_s = 0;
		std::vector<double> s_i;
		std::vector<Feature> best_feats;

		for (auto& r : roi)
		{
			std::vector<Feature> feats = extractor->extractFeatures(r.frame, n);
			n_i.push_back(feats.size());

			for (auto& f : feats)
			{
				f.column = r.x*grid_size[1] + f.column;
				f.row = r.y*grid_size[0] + f.row;
				s_i.push_back(f.score);
				best_feats.push_back(f);
			}
		}
		std_n = standardDeviation(n_i);
		std_s = standardDeviation(s_i);
		
		double _cost = std_n + std_s;

		if (_cost < cost)
		{
			best = &fr;
			cost = _cost;
			features = best_feats;
		}
	}
	init = true;

	if (verbose)
		std::cout << "Initialised using " << features.size() << 
		" features from frame #" << best->frame << "." << std::endl;
}

double Tracker::standardDeviation(std::vector<double> val)
{
	double avg = 0, std = 0;

	for (auto const& v : val)
		avg += v;
	avg /= val.size();

	for (auto const& v : val)
		std += std::pow(v - avg, 2);

	return std::sqrt(std);
}

std::vector<Tracker::GridSection> Tracker::getGridROI(Frame& fr)
{
	int gr = std::ceil((double)fr.bw.rows / (double)grid_size[0]);
	int gc = std::ceil((double)fr.bw.cols / (double)grid_size[1]);
	int fn = 0;
	std::vector<Tracker::GridSection> roi;

	for (int r = 0; r < fr.bw.rows; r += grid_size[0])
	{
		for (int c = 0; c < fr.bw.cols; c += grid_size[1])
		{
			cv::Rect rec(
				c, r, std::min((int)grid_size[1], fr.bw.cols - c),
				std::min((int)grid_size[0], fr.bw.rows - r)
			);
			roi.push_back(GridSection(fr.regionOfInterest(rec), c/grid_size[1], r/grid_size[0]));
		}
	}
	return roi;
}

double Tracker::computeFeatureCost(Frame& frame)
{
	return 0;
}

void Tracker::countTrackedFeatures()
{
	std::vector<Feature> new_feats;
	tracked_features = 0;

	for (auto& f : features)
	{
		if (f.tracked)
		{
			new_feats.push_back(f);
			tracked_features++;
		}
	}
	features = new_feats;
}