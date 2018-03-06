#include "OpenCVLucasKanadeFM.h"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

BaseFeatureMatcher::fmap OpenCVLucasKanadeFM::matchFeatures(Frame & src, Frame & next)
{
	fmap correspondences;
	std::vector<cv::Point2f> prev_points, next_points;

	for (auto const& p : src.map)
	{
		prev_points.push_back(cv::Point2f(p.first.column, p.first.row));
	}

	std::vector<uchar> status;
	std::vector<float> err;
	cv::calcOpticalFlowPyrLK(src.bw, next.bw, prev_points, next_points, status, err, cv::Size(win_size, win_size), pyr_size);

	int i = 0;

	for (auto const& p : src.map)
	{
		if (status.size() < i + 1)
			continue;
		if (status[i])
		{
			Feature f = Feature(next_points[i].x, next_points[i].y);
			next.map[f] = std::weak_ptr<Feature3D>(p.second);
			correspondences[p.first] = f;
		}
		i++;
	}
	return correspondences;
}