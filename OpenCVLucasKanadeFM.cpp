#include "OpenCVLucasKanadeFM.h"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

void OpenCVLucasKanadeFM::matchFeatures(
	Frame & src, 
	Frame & next, 
	std::vector<Feature>& feats, 
	std::vector<Feature>& new_feats
)
{
	std::vector<cv::Point2f> points, next_points;

	for (auto const& f : feats)
	{
		points.push_back(cv::Point2f(f.column, f.row));
	}

	std::vector<uchar> status;
	std::vector<float> err;
	cv::calcOpticalFlowPyrLK(src.bw, next.bw, points, next_points, status, err, cv::Size(21, 21), 4);

	for (int i = 0; i < feats.size(); i++)
	{
		Feature f(next_points[i].x, next_points[i].y);

		if (status[i])
		{
			f.tracked = true;
		}
		new_feats.push_back(f);
	}
}
