#include "Tracker.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

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

		std::vector<cv::Point> p1, p2;

		for (auto& f : features)
			p1.push_back(f.point());
		for (auto& f : new_feats)
			p2.push_back(f.point());

		cv::Mat _R, _t, mask;
		cv::Mat E = cv::findEssentialMat(p1, p2, camera, cv::RANSAC, 0.999, 1.0);
		cv::recoverPose(E, p1, p2, camera, _R, _t, mask);
		
		// possible heuristics
		//if ((_t.at<double>(2) > _t.at<double>(0)) && (_t.at<double>(2) > _t.at<double>(1)))
		//{
			R.push_back(_R);
			t.push_back(_t);
		//}
		cv::Mat roi = frame.orig(cv::Rect(0, frame.orig.rows-128, 128, 128));
		cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		double alpha = 0.75;
		cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);

		if (t.size())
		{
			for (int i = 1; i < t.size(); i++)
			{
				cv::Mat __R, __t, mask;
				__R = R[0].clone();
				__t = t[0].clone();

				for (int j = 1; j <= i; j++)
				{
					__t += (__R*t[j]);
					__R = R[j] * __R;
					cv::circle(frame.orig, cv::Point(64 + (int)__t.at<double>(0), frame.orig.rows - 8 + (int)__t.at<double>(2)), .5, cv::Scalar(0, 255, 0), 2);
				}
			}
		}
		
		/*for (int i = 0; i < new_feats.size(); i++)
		{
			if (new_feats[i].tracked)
			{
				circle(frame.orig, new_feats[i].point(), 5, cv::Scalar(0, 255, 255));
				line(frame.orig, features[i].point(), new_feats[i].point(), cv::Scalar(0, 255, 255));
			}
		}*/
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

		std::vector<GridSection> roi = getGridROI(frames[frames.size() - 1]);
		int n_grid = std::ceil((double)n / (double)roi.size());

		for (auto& r : roi)
		{
			std::vector<Feature> new_feats = extractor->extractFeatures(r.frame, n);

			for (auto& f : new_feats)
			{
				if (!f.hasNeighbor(features))
				{
					f.column = r.x*grid_size[1] + f.column;
					f.row = r.y*grid_size[0] + f.row;
					features.push_back(f);
				}
			}
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
		" features from frame #" << best->frame << std::endl;
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