#include "Tracker.h"
#include "OpenCVFASTFeatureExtractor.h"
#include "OpenCVLucasKanadeFM.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

void Tracker::start()
{
	int i = 0;

	for (auto& fn : file_names)
	{
		if (i >= stop)
			break;
		Frame frame(fn);

		if (frame.isEmpty())
			continue;

		addFrame(frame);
		i++;
	}
}

Tracker::Tracker(std::string cfg)
{
	extractor = new OpenCVFASTFeatureExtractor();
	matcher = new OpenCVLucasKanadeFM();

	std::ifstream cfg_file;
	cfg_file.open(cfg);

	if (!cfg_file) {
		throw TrackerException("Unable to open configuration file");
	}
	std::string cfg_line;
	int cfg_case = 0;

	std::string img_path, timestamp_path;
	int num_calib = 0;
	std::stringstream s_calib;

	while (std::getline(cfg_file, cfg_line)) {

		switch (cfg_case)
		{
		// Image file path
		case 0:
			cv::glob(cfg_line, file_names, false);
			break;
		// Camera identifier
		case 1:
			s_calib = std::stringstream(cfg_line);
			s_calib >> num_calib;
			break;
		// KITTI calibration file path
		case 2:
			parseCalibration(cfg_line, num_calib);
			break;
		case 3:
		// KITTI ground truth
			parsePoses(cfg_line);
			break;
		}

		cfg_case++;
	}
	cfg_file.close();
}

std::vector<std::string> Tracker::split(const std::string& str, const std::string& delim)
{
	std::vector<std::string> tokens;
	std::size_t prev = 0, pos = 0;
	do
	{
		pos = str.find(delim, prev);

		if (pos == std::string::npos)
		{
			pos = str.length();
		}
		std::string token = str.substr(prev, pos - prev);

		if (!token.empty())
		{
			tokens.push_back(token);
		}
		prev = pos + delim.length();
	}
	while (pos < str.length() && prev < str.length());

	return tokens;
}

void Tracker::parsePoses(std::string filename)
{
	std::fstream f_poses;
	f_poses.open(filename);

	if (!f_poses)
	{
		throw TrackerException("Unable to open pose file");
	}
	std::string pose;

	while (std::getline(f_poses, pose))
	{
		std::vector<std::string> s_pose = split(pose);
		int i = 0;
		cv::Mat_<double> _R(3, 3);
		cv::Mat_<double> _t(3, 1);

		for (auto const& p : s_pose)
		{
			std::stringstream ss_pose(p);
			double j;
			ss_pose >> j;

			switch (i)
			{
			case 0:
				_R.at<double>(0, 0) = j;
				break;
			case 1:
				_R.at<double>(0, 1) = j;
				break;
			case 2:
				_R.at<double>(0, 2) = j;
				break;
			case 4:
				_R.at<double>(1, 0) = j;
				break;
			case 5:
				_R.at<double>(1, 1) = j;
				break;
			case 6:
				_R.at<double>(1, 2) = j;
				break;
			case 8:
				_R.at<double>(2, 0) = j;
				break;
			case 9:
				_R.at<double>(2, 1) = j;
				break;
			case 10:
				_R.at<double>(2, 2) = j;
				break;
			case 3:
				_t.at<double>(0, 0) = j;
				break;
			case 7:
				_t.at<double>(1, 0) = j;
				break;
			case 11:
				_t.at<double>(2, 0) = j;
				break;
			}
			i++;
		}
		gt_R.push_back(_R);
		gt_t.push_back(_t);
	}
}

void Tracker::parseCalibration(std::string filename, int num_calib)
{
	std::fstream f_calib;
	f_calib.open(filename);

	if (!f_calib)
	{
		throw TrackerException("Unable to open calibration file");
	}
	std::string calib;
	int i = 0;

	while (std::getline(f_calib, calib))
	{
		if (i == num_calib)
		{
			int k = 0;
			int pos = 0;
			std::string token;

			while ((pos = calib.find(" ")) != std::string::npos)
			{
				std::stringstream todouble(calib.substr(0, pos));
				double j;
				todouble >> j;
				calib.erase(0, pos + 1);

				switch (k)
				{
				case 1:
					camera.at<double>(0, 0) = j;
					break;
				case 2:
					camera.at<double>(0, 1) = j;
					break;
				case 3:
					camera.at<double>(0, 2) = j;
					break;
				case 5:
					camera.at<double>(1, 0) = j;
					break;
				case 6:
					camera.at<double>(1, 1) = j;
					break;
				case 7:
					camera.at<double>(1, 2) = j;
					break;
				case 9:
					camera.at<double>(2, 0) = j;
					break;
				case 10:
					camera.at<double>(2, 1) = j;
					break;
				case 11:
					camera.at<double>(2, 2) = j;
					break;
				}
				k++;
			}
		}
		i++;
	}
	f_calib.close();
}

double Tracker::tock()
{
	if (ticktock.empty())
		return 0;
	double tock = ticktock.back();
	ticktock.pop_back();
	return (cv::getTickCount() - tock) / cv::getTickFrequency();
}

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
		std::cout << "--- Frame #" << frame.frame << " ---" << std::endl;
		Frame& src = frames[frames.size() - 2];
		Frame& next = frames[frames.size() - 1];
		std::vector<Feature> new_feats;

		tick();
		matcher->matchFeatures(src, next, features, new_feats);
		std::cout << tock() << " seconds for feature matching ";

		std::vector<cv::Point> p1, p2;

		for (auto& f : features)
			p1.push_back(f.point());
		for (auto& f : new_feats)
			p2.push_back(f.point());

		tick();
		cv::Mat _R, _t, mask;
		cv::Mat E = cv::findEssentialMat(p1, p2, camera, cv::RANSAC, 0.99, 1, mask);
		std::cout << tock() << " seconds for finding E ";
		tick();
		cv::recoverPose(E, p1, p2, camera, _R, _t, mask);
		std::cout << tock() << " seconds for pose estimation " << std::endl;

		// possible heuristics
		//if ((_t.at<double>(2) > _t.at<double>(0)) && (_t.at<double>(2) > _t.at<double>(1)))
		//{
			R.push_back(_R);
			t.push_back(_t);
		//}

		if (t.size())
		{
			for (int i = 1; i < t.size(); i++)
			{
				cv::Mat __R, __t, mask;
				__R = R[0].clone();
				__t = t[0].clone();

				for (int j = 1; j <= i; j++)
				{
					cv::Mat dist = gt_t[j] - gt_t[j - 1];
					double scale = std::sqrt(
						std::pow(dist.at<double>(0), 2) +
						std::pow(dist.at<double>(1), 2) +
						std::pow(dist.at<double>(2), 2));

					__t += scale*(__R*t[j]);
					__R = R[j] * __R;
					cv::circle(
						map,
						cv::Point(map.cols / 2 + (int)__t.at<double>(0), map.rows - 8 + (int)__t.at<double>(2)),
						.5,
						cv::Scalar(0, 255, 0),
						2);

					cv::circle(
						map,
						cv::Point(map.cols / 2 + (int)gt_t[j].at<double>(0), map.rows - 8 - (int)gt_t[j].at<double>(2)),
						.5,
						cv::Scalar(0, 0, 255),
						2);
				}
			}
		}

		if (fancy_video)
		{
			cv::Mat roi = frame.orig(cv::Rect(0, frame.orig.rows - 256, 256, 256));
			cv::Mat colour = cv::Mat::zeros(roi.size(), frame.orig.type());
			cv::resize(map, colour, colour.size());
			double alpha = 0.75;
			cv::addWeighted(colour, alpha, roi, 1.0 - alpha, 0.0, roi);

			/*for (int i = 0; i < new_feats.size(); i++)
			{
			if (new_feats[i].tracked)
			{
			circle(frame.orig, new_feats[i].point(), 5, cv::Scalar(0, 255, 255));
			line(frame.orig, features[i].point(), new_feats[i].point(), cv::Scalar(0, 255, 255));
			}
			}*/
		}
		features = new_feats;
		
		cv::imshow("map", map);
		cv::imshow("test", frame.orig);
		cv::waitKey(10);
	}
	countTrackedFeatures();

	if (init && tracked_features + tracked_features_tol < min_tracked_features)
	{
		int n = min_tracked_features - tracked_features;
		tick();
		if (verbose)
			std::cout << "Trying to find " << n << " new features" << 
			" in frame #" << (frames.size() - 1) << std::endl;

		std::vector<Feature> new_feats = extractor->extractFeatures(frame, n);

		std::vector<GridSection> roi = getGridROI(frames[frames.size() - 1]);
		int n_grid = std::ceil((double)n / (double)roi.size());

		for (auto& r : roi)
		{
			std::vector<Feature> new_feats = extractor->extractFeatures(r.frame, n_grid);

			for (auto& f : new_feats)
			{
				if (features.size() >= min_tracked_features + tracked_features_tol)
					break;
				//if (!f.hasNeighbor(features))
				//{
					f.column = r.x*grid_size[1] + f.column;
					f.row = r.y*grid_size[0] + f.row;
					features.push_back(f);
				//}
			}
		}
		if (verbose)
			std::cout << "Feature extraction took " << tock() << " seconds";
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
		double n = (min_tracked_features + tracked_features_tol) / roi.size();
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

void Tracker::countTrackedFeatures()
{
	std::vector<Feature> new_feats;
	tracked_features = 0;

	for (auto& f : features)
	{
		if (tracked_features >= min_tracked_features + tracked_features_tol)
			break;
		if (f.tracked)
		{
			new_feats.push_back(f);
			tracked_features++;
		}
	}
	features = new_feats;
}