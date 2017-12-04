#include "Tracker.h"
#include <iostream>

void Tracker::addFrame(Frame& frame)
{
	if (frames.size() >= init_frames && !init)
	{
		initialise();
	}
	frame.frame = frames.size();
	frames.push_back(frame);
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
		std::vector<Frame> roi = getGridROI(fr);
		double n = init_features / roi.size();
		double std_n = 0;
		std::vector<double> n_i;
		double std_s = 0;
		std::vector<double> s_i;
		std::vector<Feature> best_feats;

		for (auto& r : roi)
		{
			std::vector<Feature> feats = extractor->extractFeatures(r, n);
			n_i.push_back(feats.size());

			for (auto& f : feats)
			{
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

std::vector<Frame> Tracker::getGridROI(Frame& fr)
{
	int gr = std::ceil((double)fr.bw.rows / (double)grid_size[0]);
	int gc = std::ceil((double)fr.bw.cols / (double)grid_size[1]);
	int fn = 0;
	std::vector<Frame> roi;

	for (int r = 0; r < fr.bw.rows; r += grid_size[0])
	{
		for (int c = 0; c < fr.bw.cols; c += grid_size[1])
		{
			cv::Rect rec(
				c, r, std::min((int)grid_size[1], fr.bw.cols - c),
				std::min((int)grid_size[0], fr.bw.rows - r)
			);
			roi.push_back(fr.regionOfInterest(rec));
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
	tracked_features = 0;

	for (auto& f : features)
	{
		if (f.tracked)
		{
			tracked_features++;
		}
	}
}