#include "OpenCVEPnPSolver.h"
#include <opencv2/calib3d.hpp>

void OpenCVEPnPSolver::solvePnP(Frame& src, Frame& next, cv::Mat& R_out, cv::Mat& t_out)
{
	int j = src.frame;
	std::vector<cv::Point3f> obj_points;
	std::vector<cv::Point2f> img_points;
	cv::Mat _R_rod;
	cv::Rodrigues(R_out, _R_rod);
	std::vector<std::weak_ptr<Feature3D>> local_feats3d;

	for (auto& p : src.map)
	{
		if (p.second.expired())
			continue;
		std::shared_ptr<Feature3D> f3d = p.second.lock();

		if (src.feat_corr[p.first].expired())
			continue;
		std::shared_ptr<Feature> f = src.feat_corr[p.first].lock();
		next.map[f] = std::weak_ptr<Feature3D>(f3d);
		f3d->transformInv(tracker->R[j], tracker->t[j]);

		cv::Point3f p3f = f3d->getPoint();
		p3f.z *= -1;

		obj_points.push_back(p3f);
		img_points.push_back(f->getPoint());

		f3d->transform(tracker->R[j], tracker->t[j]);
		local_feats3d.push_back(f3d);
	}
	std::vector<int> inliers;
	cv::solvePnPRansac(obj_points, img_points, tracker->camera, 
		cv::Mat(), _R_rod, t_out, true, 100, 8, .99, inliers);
	cv::Rodrigues(_R_rod, R_out);

	// Removing RANSAC outliers
	for (int i = 0; i < obj_points.size(); i++)
	{
		if (std::find(inliers.begin(), inliers.end(), i) == inliers.end())
		{
			if (local_feats3d[i].expired())
				continue;
			std::shared_ptr<Feature3D> f3d = local_feats3d[i].lock();
			tracker->feats3d.erase(std::find(tracker->feats3d.begin(), tracker->feats3d.end(), f3d));
		}
	}
}