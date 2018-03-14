#include "OpenCVFivePointTri.h"
#include <opencv2/calib3d.hpp>
#include "OdometryPipeline.h"

void OpenCVFivePointTri::triangulate(Frame & src, Frame & next, cv::Mat & R_out, cv::Mat & t_out)
{
	int j = src.frame;
	std::vector<cv::Point> p1, p2;
	std::vector<std::shared_ptr<Feature>> p1_ptr, p2_ptr;

	cv::Mat mask, tri;

	for (auto& p : src.feat_corr)
	{
		if (p.first.expired() || p.second.expired())
			continue;
		std::shared_ptr<Feature> fst = p.first.lock();
		std::shared_ptr<Feature> sec = p.second.lock();
		p1.push_back(fst->getPoint());
		p2.push_back(sec->getPoint());
		p1_ptr.push_back(fst);
		p2_ptr.push_back(sec);
	}
	cv::Mat E = cv::findEssentialMat(p1, p2, tracker->camera, cv::RANSAC, 0.99, 1, mask);

	cv::recoverPose(E, p1, p2, tracker->camera, R_out, t_out, HUGE_VAL, mask, tri);

	cv::Mat dist = tracker->gt_t[j + tracker->init_offset + 1] - tracker->gt_t[j + tracker->init_offset];
	tracker->scale = std::sqrt(
		std::pow(dist.at<double>(0), 2) +
		std::pow(dist.at<double>(1), 2) +
		std::pow(dist.at<double>(2), 2));

	t_out = tracker->scale*t_out;

	for (int i = 0; i < tri.cols; i++)
	{
		// Removing RANSAC outliers
		if (!mask.at<char>(i))
			continue;
		std::shared_ptr<Feature3D> f3d_ptr = std::make_shared<Feature3D>(
			tracker->scale*tri.at<double>(0, i) / tri.at<double>(3, i),
			tracker->scale*tri.at<double>(1, i) / tri.at<double>(3, i),
			tracker->scale*tri.at<double>(2, i) / tri.at<double>(3, i) * -1);

		if (f3d_ptr->getPoint().z < 0)
		{
			f3d_ptr->transform(tracker->R[j], tracker->t[j]);
			tracker->feats3d.push_back(f3d_ptr);
			next.map[p2_ptr[i]] = std::weak_ptr<Feature3D>(f3d_ptr);
			src.map[p1_ptr[i]] = std::weak_ptr<Feature3D>(f3d_ptr);
		}
	}
}
