#include "OdometryPipeline.h"
#include "OpenCVFASTFeatureExtractor.h"
#include "OpenCVLucasKanadeFM.h"
#include "OpenCVGoodFeatureExtractor.h"
#include "OpenCVEPnPSolver.h"
#include "OpenCVFivePointTri.h"
#include "CeresBundleAdjustment.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "Feature3D.h"
#include "ProjectionResidual.h"
#include <ceres/ceres.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/misc_api.h>

#include "kNNFeatureMatcher.h"
#include "ShiTomasiFeatureExtractor.h"

dlib::mutex frame_mutex;
dlib::signaler frame_signaler(frame_mutex);

OdometryPipeline::OdometryPipeline(std::string cfg_path) : job_pipe(605)
{
	std::ifstream cfg_file;
	cfg_file.open(cfg_path);

	if (!cfg_file)
		throw OdometryPipelineException("Unable to open configuration file");

	std::string cfg_line, name, value;
	int cfg_case = 0;
	int div;
	std::map<std::string, std::string> cfg;

	while (std::getline(cfg_file, cfg_line))
	{
		cfg_line = trimString(cfg_line);

		if (!cfg_line.length() || cfg_line[0] == '#' || cfg_line[0] == ';' || cfg_line[0] == '[')
			continue;
		div = cfg_line.find('=');
		name = trimString(cfg_line.substr(0, div));
		value = trimString(cfg_line.substr(div + 1));
		cfg[name] = value;
	}
	fancy_video = std::stoi(cfg["fancy_video"]);
	verbose = std::stoi(cfg["verbose"]);
	min_tracked_features = std::stoi(cfg["min_tracked_features"]);
	tracked_features_tol = std::stoi(cfg["tracked_features_tol"]);
	init_frames = std::stoi(cfg["init_frames"]);
	stop = std::stoi(cfg["frames"]);
	bundle_size = std::stoi(cfg["bundle_size"]);
	ba_iterations = std::stoi(cfg["max_iterations"]);
	video_path = cfg["video_path"];

	cv::glob(cfg["image_dir"], file_names, false);
	parseCalibration(cfg["camera_calibration"], std::stoi(cfg["camera"]));
	parsePoses(cfg["poses"]);

	cfg_file.close();

	extractor = new OpenCVGoodFeatureExtractor();
	matcher = new OpenCVLucasKanadeFM();
	pnpsolver = new OpenCVEPnPSolver(this);
	triangulator = new OpenCVFivePointTri(this);
	ba = new CeresBundleAdjustment(this);
}

double OdometryPipeline::median(std::vector<double> &v)
{
	size_t n = v.size() / 2;
	std::nth_element(v.begin(), v.begin() + n, v.end());
	if (!n)
		return 0;
	return v[n];
}

double OdometryPipeline::tock()
{
	if (ticktock.empty())
		return 0;
	double tock = ticktock.back();
	ticktock.pop_back();
	return (cv::getTickCount() - tock) / cv::getTickFrequency();
}

void OdometryPipeline::drawCross(const int radius, const cv::Point& pos, const cv::Scalar& color, cv::Mat& dst, int thickness)
{
	cv::Point a1 = cv::Point(pos.x - radius + 1, pos.y);
	cv::Point a2 = cv::Point(pos.x + radius - 1, pos.y);
	cv::line(dst, a1, a2, color, thickness);

	a1 = cv::Point(pos.x, pos.y - radius + 1);
	a2 = cv::Point(pos.x, pos.y + radius - 1);
	cv::line(dst, a1, a2, color, thickness);
}

void OdometryPipeline::drawMap(Frame& fr)
{
	int j = t.size() - 1;
	map = cv::Mat::zeros(512, 512, CV_8UC3);

	// Drawing 3D feature points
	for (auto& p : fr.map)
	{
		if (p.second.expired())
			continue;
		auto f3d = p.second.lock();
		auto f = p.first;

		cv::Scalar color;

		if (f->column > fr.orig.cols / 2)
			color = cv::Scalar(255, 0, 255);
		else
			color = cv::Scalar(255, 255, 0);

		drawCross(3, cv::Point(f->column, f->row), color, fr.orig, 1);

		cv::circle(map, cv::Point(map.cols / 2 + f3d->getPoint().x, map.rows / 1.2 + f3d->getPoint().z), 1, color, -1);
	}

	// Drawing rectangle representing position and orientation

	int x = map.cols / 2 + (int)t[j].at<double>(0);
	int y = map.rows / 1.2 + (int)t[j].at<double>(2);
	cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(10, 15), calcYRotation(R[j]) / 3.1416 * 180);
	cv::Point2f vertices[4];
	rRect.points(vertices);

	for (int i = 0; i < 4; i++)
		cv::line(map, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0));

	// Same as above but for the ground truth
	x = map.cols / 2 + (int)gt_t[j].at<double>(0);
	y = map.rows / 1.2 - (int)gt_t[j].at<double>(2);
	rRect = cv::RotatedRect(cv::Point2f(x, y), cv::Size2f(10, 15), calcYRotation(gt_R[j], true) / 3.1416 * 180);
	rRect.points(vertices);

	for (int i = 0; i < 4; i++)
		cv::line(map, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255));

	// Tracing the path of the calculated trajectory as well as the ground truth on the map
	for (int i = init_offset; i <= j; i++)
	{
		int x = map.cols / 2 + (int)t[i].at<double>(0);
		int y = map.rows / 1.2 + (int)t[i].at<double>(2);
		cv::circle(
			map,
			cv::Point(x, y),
			.5,
			cv::Scalar(0, 255, 0),
			2);

		cv::circle(
			map,
			cv::Point(map.cols / 2 + (int)gt_t[i].at<double>(0), map.rows / 1.2 - (int)gt_t[i].at<double>(2)),
			.5,
			cv::Scalar(0, 0, 255),
			2);
	}
}

void OdometryPipeline::motionHeuristics(cv::Mat& _R, cv::Mat& _t, int j)
{
	if (_t.at<double>(2) < 0 && calcYRotation(_R) < 3.1415/8
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
		_t = .5*t_s[j];
		_R = R_s[j];

		t_s.push_back(t_s[j].clone());
		R_s.push_back(R_s[j].clone());

		_t = R[j] * t_s[j] + t[j];
		_R = R_s[j ] * R[j];

		/*_t = cv::Mat::zeros(3, 1, CV_64F);
		_t.at<double>(2) = 0;
		_R = cv::Mat::eye(3, 3, CV_64F);

		t_s.push_back(_t.clone());
		R_s.push_back(_R.clone());

		_t = R[j] * _t + t[j];
		_R = _R*R[j];*/
	}
	t.push_back(_t);
	R.push_back(_R);
}

void OdometryPipeline::featureExtractionThread()
{
	for (int i = init_offset + 1; i < file_names.size(); i++)
	{
		if (i >= stop)
			break;
		Frame frame(file_names[i]);

		if (frame.isEmpty())
			continue;
		addFrame(frame);

		dlib::auto_mutex locker(frame_mutex);
		if (frame.frame < 2)
			continue;
		Job j;
		j.frame = frame.frame-2;
		locker.unlock();
		job_pipe.enqueue(j);
	}
	job_pipe.wait_until_empty();
	job_pipe.disable();
}

void OdometryPipeline::poseEstimationThread()
{
	Job j;
	while (job_pipe.dequeue(j))
	{
		dlib::auto_mutex locker(frame_mutex);
		Frame src = *(frames[j.frame]);
		Frame next = *(frames[j.frame + 1]);
		locker.unlock();
		estimatePose(src, next);
	}
}

void OdometryPipeline::startPipeline()
{
	for (int i = 0; i < init_frames; i++)
		frames.push_back(std::make_shared<Frame>(Frame(file_names[i])));

	initialise();
	cv::Mat _R = ((cv::Mat_<double>(3, 3)) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	cv::Mat _t = ((cv::Mat_<double>(3, 1)) << 0, 0, 0);
	R.push_back(_R);
	t.push_back(_t);
	R_s.push_back(_R);
	t_s.push_back(_t);

	register_thread(*this, &OdometryPipeline::featureExtractionThread);
	register_thread(*this, &OdometryPipeline::poseEstimationThread);
	start();
	wait();

	// Display 3d point cloud
	std::vector<dlib::perspective_window::overlay_dot> points;
	dlib::rand rnd;

	std::vector<double> x, y, z;
	for (auto& f3d : feats3d)
	{
		x.push_back(f3d->getPoint().x);
		y.push_back(f3d->getPoint().y);
		z.push_back(f3d->getPoint().z);
	}
	double med_x = median(x), med_y = median(y), med_z = median(z);

	for (auto& f3d : feats3d)
	{
		// Skimming far-away points so camera doesn't zoom out as much
		if (std::abs(f3d->getPoint().x) > 4 * std::abs(med_x) ||
			std::abs(f3d->getPoint().y) > 4 * std::abs(med_y) ||
			std::abs(f3d->getPoint().z) > 4 * std::abs(med_z))
			continue;
		dlib::vector<double> val(f3d->getPoint().x, f3d->getPoint().y, f3d->getPoint().z);
		points.push_back(dlib::perspective_window::overlay_dot(val));
	}
	dlib::perspective_window win;
	win.clear_overlay();
	win.add_overlay(points);
	win.set_size(512, 512);
	win.set_title("3D point cloud");
	win.wait_until_closed();
}

void OdometryPipeline::addFrame(Frame& frame)
{
	frame.frame = frames.size();
	dlib::auto_mutex locker(frame_mutex);
	tick();
	BaseFeatureMatcher::fmap feat_corr = matcher->matchFeatures(*(frames[frame.frame-1]), frame);
	frames[frame.frame - 1]->feat_corr = feat_corr;
	locker.unlock();

	if (verbose)
		std::cout << tock() << " seconds for feature matching in frame #" << frame.frame << std::endl;
	tick();

	if (feat_corr.size() < tracked_features_tol)
	{
		if (verbose)
			std::cout << "Trying to find " << min_tracked_features << " new features" <<
			" in frame #" << (frames.size() - 1) << std::endl;
		dlib::auto_mutex locker2(frame_mutex);
		std::vector<GridSection> roi = getGridROI(*(frames[frames.size() - 1]));
		locker2.unlock();
		int n_grid = std::ceil((double)min_tracked_features / (double)roi.size());

		for (auto& r : roi)
		{
			std::vector<Feature> new_feats = extractor->extractFeatures(r.frame, n_grid);

			for (auto& f : new_feats)
			{
				if (!frame.hasNeighbor(f))
				{
					f.column = r.x*grid_size[1] + f.column;
					f.row = r.y*grid_size[0] + f.row;
					frame.map[std::make_shared<Feature>(f)] = std::weak_ptr<Feature3D>();
				}
			}
		}
		if (verbose)
			std::cout << "Feature extraction took " << tock() << " seconds" << std::endl;
	}
	dlib::auto_mutex locker3(frame_mutex);
	frames.push_back(std::make_shared<Frame>(frame));
}

void OdometryPipeline::estimatePose(Frame& src, Frame &next)
{
	tick();
	int j = src.frame;
	cv::Mat _R = R[j].clone(), _t = t[j].clone();

	if (src.count3DPoints() >= tracked_features_tol)
	{
		pnpsolver->solvePnP(src, next, _R, _t);
	}
	else
	{
		tick();
		triangulator->triangulate(src, next, _R, _t);
		init3d = true;

		if (verbose)
			std::cout << tock() << " seconds for triangulating points." << std::endl;
		tick();
	}
	motionHeuristics(_R, _t, j);

	dlib::auto_mutex locker(frame_mutex);
	frames[src.frame] = std::make_shared<Frame>(src);
	frames[next.frame] = std::make_shared<Frame>(next);
	locker.unlock();

	if (verbose)
		std::cout << tock() << " seconds for pose estimation in frame #" << src.frame << std::endl;

	if (bundle_size && src.frame && src.frame % (bundle_size / 3 * 2) == 0)
	{
		dlib::auto_mutex locker_ba(frame_mutex);
		ba->apply(next);
		locker_ba.unlock();
	}
	drawMap(src);

	if (fancy_video)
	{
		cv::Mat roi = src.orig(cv::Rect(0, src.orig.rows - src.orig.rows, src.orig.rows, src.orig.rows));
		cv::Mat colour = cv::Mat::zeros(roi.size(), src.orig.type());
		cv::resize(map, colour, colour.size());
		double alpha = 0.75;
		cv::addWeighted(colour, alpha, roi, 1.0 - alpha, 0.0, roi);
	}
	cv::imshow("map", map);
	cv::imshow("test", src.orig);
	cv::waitKey(10);
}

void OdometryPipeline::initialise()
{
	int i = 0;

	if (verbose)
		std::cout << "Initialising..." << std::endl;

	Frame best = *(frames[0]);
	double cost = HUGE_VAL;

	for (auto& fr : frames)
	{
		std::vector<GridSection> roi = getGridROI(*fr);
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
				fr->map[std::make_shared<Feature>(f)] = std::weak_ptr<Feature3D>();
			}
		}
		std_n = standardDeviation(n_i);
		std_s = standardDeviation(s_i);
		
		double _cost = std_n + std_s;

		if (_cost < cost)
		{
			fr->frame = 0;
			best = *fr;
			cost = _cost;
			init_offset = i;
		}
		i++;
	}
	init = true;
	frames.clear();
	frames.push_back(std::make_shared<Frame>(best));

	if (verbose)
		std::cout << "Initialised using " << best.map.size() << 
		" features from frame #" << init_offset << std::endl;
}

std::string OdometryPipeline::trimString(std::string const& str, char const* delim)
{
	std::string dest(str);
	std::string::size_type index = dest.find_first_not_of(delim);

	if (index != std::string::npos)
		dest.erase(0, index);
	else
		dest.erase();
	index = dest.find_last_not_of(delim);

	if (index != std::string::npos)
		dest.erase(++index);
	return dest;
}

std::vector<std::string> OdometryPipeline::split(const std::string& str, const std::string& delim)
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

void OdometryPipeline::parsePoses(std::string filename)
{
	std::fstream f_poses;
	f_poses.open(filename);

	if (!f_poses)
		throw OdometryPipelineException("Unable to open pose file");

	std::string pose;
	int k = 0;

	while (std::getline(f_poses, pose) && k < stop)
	{
		k++;
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

void OdometryPipeline::parseCalibration(std::string filename, int num_calib)
{
	std::fstream f_calib;
	f_calib.open(filename);

	if (!f_calib)
		throw OdometryPipelineException("Unable to open calibration file");

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

double OdometryPipeline::standardDeviation(std::vector<double> val)
{
	double avg = 0, std = 0;

	for (auto const& v : val)
		avg += v;
	avg /= val.size();

	for (auto const& v : val)
		std += std::pow(v - avg, 2);

	return std::sqrt(std);
}

std::vector<OdometryPipeline::GridSection> OdometryPipeline::getGridROI(Frame& fr)
{
	int gr = std::ceil((double)fr.bw.rows / (double)grid_size[0]);
	int gc = std::ceil((double)fr.bw.cols / (double)grid_size[1]);
	int fn = 0;
	std::vector<OdometryPipeline::GridSection> roi;

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