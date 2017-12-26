#include "Tracker.h"
#include "OpenCVFASTFeatureExtractor.h"
#include "OpenCVLucasKanadeFM.h"
#include "OpenCVGoodFeatureExtractor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "Feature3D.h"
#include "ProjectionResidual.h"
#include <ceres/ceres.h>

void Tracker::start()
{
	for (int i = 0; i < init_frames; i++)
	{
		frames.push_back(Frame(file_names[i]));
	}
	initialise();
	cv::Mat _R = ((cv::Mat_<double>(3, 3)) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	cv::Mat _t = ((cv::Mat_<double>(3, 1)) << 0, 0, 0);
	R.push_back(_R);
	t.push_back(_t);

	for (int i = init_offset + 1; i < file_names.size(); i++)
	{
		if (i >= stop)
			break;
		Frame frame(file_names[i]);

		if (frame.isEmpty())
			continue;

		addFrame(frame);
	}
}

double Tracker::tock()
{
	if (ticktock.empty())
		return 0;
	double tock = ticktock.back();
	ticktock.pop_back();
	return (cv::getTickCount() - tock) / cv::getTickFrequency();
}

void Tracker::drawMap()
{
	int j = t.size() - 1;
	int k = frames.size() - 1;
	map = cv::Mat::zeros(512, 512, CV_8UC3);

	Frame* fr = currentFrame();

	// Drawing 3D feature points
	for (auto& p : fr->map)
	{
		if (p.second.expired())
			continue;
		auto f3d = p.second.lock();
		auto f = &p.first;

		cv::Scalar color;

		if (f->column > fr->orig.cols / 2)
			color = cv::Scalar(255, 0, 255);
		else
			color = cv::Scalar(255, 255, 0);

		cv::circle(fr->orig, cv::Point(f->column, f->row), 3, color, .5);

		cv::circle(map, cv::Point(map.cols / 2 + f3d->getPoint().x, map.rows / 2 + f3d->getPoint().z), 1, color, -1);
	}

	// Drawing rectangle representing position and orientation
	int x = map.cols / 2 + (int)t[j].at<double>(0);
	int y = map.rows / 2 + (int)t[j].at<double>(2);
	cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(10, 15), std::acos(R[j].at<double>(0, 0)) / 3.1416 * 180);
	cv::Point2f vertices[4];
	rRect.points(vertices);

	for (int i = 0; i < 4; i++)
		cv::line(map, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0));

	// Same as above but for the ground truth
	x = map.cols / 2 + (int)gt_t[j].at<double>(0);
	y = map.rows / 2 - (int)gt_t[j].at<double>(2);
	rRect = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(10, 15), std::acos(gt_R[j].at<double>(0, 0)) / 3.1416 * 180);
	rRect.points(vertices);

	for (int i = 0; i < 4; i++)
		cv::line(map, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255));

	// Tracing the path of the calculated trajectory as well as the ground truth on the map
	for (int i = init_offset; i <= j; i++)
	{
		int x = map.cols / 2 + (int)t[i].at<double>(0);
		int y = map.rows / 2 + (int)t[i].at<double>(2);
		cv::circle(
			map,
			cv::Point(x, y),
			.5,
			cv::Scalar(0, 255, 0),
			2);

		cv::circle(
			map,
			cv::Point(map.cols / 2 + (int)gt_t[i].at<double>(0), map.rows / 2 - (int)gt_t[i].at<double>(2)),
			.5,
			cv::Scalar(0, 0, 255),
			2);
	}
}

void Tracker::motionHeuristics(cv::Mat& _R, cv::Mat& _t, int j)
{
	if (_t.at<double>(2) < 0 && _R.at<double>(0, 0) > .0
		&& _R.at<double>(1, 1) > .0 && _R.at<double>(2, 2) > .0
		&& std::abs(_t.at<double>(2)) > std::max(std::abs(_t.at<double>(0)), std::abs(_t.at<double>(1)))
		&& std::abs(_t.at<double>(2)) < 2 * scale)
	{
		t_s.push_back(_t.clone());
		R_s.push_back(_R.clone());

		_t = R[j] * _t + t[j];
		_R = _R*R[j];
	}
	else
	{
		if (verbose)
			std::cout << "Using heuristic motion" << std::endl;
		_t = .5*t_s[j - 1];
		_R = R_s[j - 1];

		t_s.push_back(t_s[j - 1].clone());
		R_s.push_back(R_s[j - 1].clone());

		_t = R[j] * t_s[j - 1] + t[j];
		_R = R_s[j - 1] * R[j];

		/*_t = cv::Mat::zeros(3, 1, CV_64F);
		_t.at<double>(2) = -scale;
		_R = cv::Mat::eye(3, 3, CV_64F);

		t_s.push_back(_t.clone());
		R_s.push_back(_R.clone());

		_t = R[j] * _t + t[j];
		_R = _R*R[j];*/
	}
}

void Tracker::addFrame(Frame& frame)
{
	frame.frame = frames.size();
	frames.push_back(frame);

	if (verbose)
		std::cout << "--- Frame #" << frame.frame << " ---" << std::endl;

	Frame& src = frames[frames.size() - 2];
	Frame& next = frames[frames.size() - 1];
	std::vector<Feature> new_feats;

	tick();
	BaseFeatureMatcher::fmap feat_corr =  matcher->matchFeatures(src, next);

	if (verbose)
		std::cout << tock() << " seconds for feature matching ";

	/*std::vector<cv::Point> p1, p2;
	std::vector<Feature> corr;

	for (int i = 0; i < features.size(); i++)
	{
		if (new_feats[i].tracked)
		{
			p1.push_back(features[i].point());
			p2.push_back(new_feats[i].point());
			corr.push_back(features[i]);

			features[i].row = new_feats[i].row;
			features[i].column = new_feats[i].column;
		}
		else
			features[i].tracked = false;
	}*/
	tick();

	int j = t.size() - 1;
	cv::Mat _R = R[j].clone(), _t = t[j].clone(), mask, tri;

	if (j > 1)
	{
		cv::Mat dist = gt_t[j] - gt_t[j - 1];
		scale = std::sqrt(
			std::pow(dist.at<double>(0), 2) +
			std::pow(dist.at<double>(1), 2) +
			std::pow(dist.at<double>(2), 2));
	}
	else
	{
		scale = 1;
	}

	if (true && src.count3DPoints() >= tracked_features_tol)
	{
		std::vector<cv::Point3f> obj_points;
		std::vector<cv::Point2f> img_points;
		cv::Mat _R_rod;
		cv::Rodrigues(_R, _R_rod);

		for (auto& p : src.map)
		{
			if (p.second.expired())
				continue;
			std::shared_ptr<Feature3D> f3d = p.second.lock();
			Feature f = next.get2DFeature(p.second);
			f3d->transformInv(R[j], t[j]);

			cv::Point3f p3f = f3d->getPoint();
			p3f.z *= -1;

			obj_points.push_back(p3f);
			img_points.push_back(f.getPoint());

			f3d->transform(R[j], t[j]);
		}
		cv::solvePnPRansac(obj_points, img_points, camera, cv::Mat(), _R_rod, _t, true);
		cv::Rodrigues(_R_rod, _R);

		motionHeuristics(_R, _t, j);
	}
	else
	{
		tick();

		std::vector<cv::Point> p1, p2;

		for (auto& p : feat_corr)
		{
			Feature f = p.first;
			p1.push_back(f.getPoint());
			p2.push_back(p.second.getPoint());
		}
		cv::Mat E = cv::findEssentialMat(p1, p2, camera, cv::RANSAC, 0.99, 1, mask);

		if (verbose)
			std::cout << tock() << " seconds for finding E ";
		tick();
		cv::recoverPose(E, p1, p2, camera, _R, _t, HUGE_VAL, mask, tri);

		_t = scale*_t;
		motionHeuristics(_R, _t, j);

		for (int i = 0; i < tri.cols; i++)
		{
			std::shared_ptr<Feature3D> f3d_ptr = std::make_shared<Feature3D>(
				scale*tri.at<double>(0, i) / tri.at<double>(3, i),
				scale*tri.at<double>(1, i) / tri.at<double>(3, i),
				scale*tri.at<double>(2, i) / tri.at<double>(3, i) * -1);

			if (f3d_ptr->getPoint().z < 0)
			{
				f3d_ptr->transform(R[j], t[j]);
				feats3d.push_back(f3d_ptr);
				next.map[Feature(p2[i].x, p2[i].y)] = std::weak_ptr<Feature3D>(f3d_ptr);
			}
		}
		init3d = true;
	}

	if (verbose)
		std::cout << tock() << " seconds for pose estimation " << std::endl;

	t.push_back(_t);
	R.push_back(_R);

	drawMap();

	int n3d = currentFrame()->count3DPoints();

	if (n3d < tracked_features_tol)
	{
		int n = min_tracked_features - n3d;
		tick();
		if (verbose)
			std::cout << "Trying to find " << n << " new features" << 
			" in frame #" << (frames.size() - 1) << std::endl;

		std::vector<GridSection> roi = getGridROI(frames[frames.size() - 1]);
		int n_grid = std::ceil((double)n / (double)roi.size());

		for (auto& r : roi)
		{
			std::vector<Feature> new_feats = extractor->extractFeatures(r.frame, n_grid);

			for (auto& f : new_feats)
			{
				//if (!f.hasNeighbor(currentFrame()))
				{
					f.column = r.x*grid_size[1] + f.column;
					f.row = r.y*grid_size[0] + f.row;
					currentFrame()->map[f] = std::weak_ptr<Feature3D>();
				}
			}
		}
		if (verbose)
			std::cout << "Feature extraction took " << tock() << " seconds" << std::endl;
	}

	if (fancy_video)
	{
		cv::Mat roi = frame.orig(cv::Rect(0, frame.orig.rows - frame.orig.rows, frame.orig.rows, frame.orig.rows));
		cv::Mat colour = cv::Mat::zeros(roi.size(), frame.orig.type());
		cv::resize(map, colour, colour.size());
		double alpha = 0.75;
		cv::addWeighted(colour, alpha, roi, 1.0 - alpha, 0.0, roi);
	}
	cv::imshow("map", map);
	cv::imshow("test", currentFrame()->orig);
	cv::waitKey(10);
}

void Tracker::initialise()
{
	int i = 0;

	if (verbose)
		std::cout << "Initialising..." << std::endl;

	Frame best = frames[0];
	double cost = HUGE_VAL;

	for (auto& fr : frames)
	{
		std::vector<GridSection> roi = getGridROI(fr);
		double n = min_tracked_features / roi.size();
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
				fr.map[f] = std::weak_ptr<Feature3D>();
			}
		}
		std_n = standardDeviation(n_i);
		std_s = standardDeviation(s_i);
		
		double _cost = std_n + std_s;

		if (_cost < cost)
		{
			fr.frame = 0;
			best = fr;
			cost = _cost;
			init_offset = i;
			//features = best_feats;
		}
		i++;
	}
	init = true;
	frames.clear();
	frames.push_back(best);

	if (verbose)
		std::cout << "Initialised using " << best.map.size() << 
		" features from frame #" << init_offset - 1 << std::endl;
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
	} while (pos < str.length() && prev < str.length());

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