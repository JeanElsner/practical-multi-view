#include "CeresBundleAdjustment.h"
#include "ProjectionResidual.h"
#include <opencv2/calib3d.hpp>

void CeresBundleAdjustment::apply(Frame & f)
{
	int fn = (int)f.frame + 1;
	int n = std::min(tracker->bundle_size, fn);

	double _camera[] = {
		tracker->camera.at<double>(0, 0), tracker->camera.at<double>(0, 1), tracker->camera.at<double>(0, 2),
		tracker->camera.at<double>(1, 0), tracker->camera.at<double>(1, 1), tracker->camera.at<double>(1, 2),
		tracker->camera.at<double>(2, 0), tracker->camera.at<double>(2, 1), tracker->camera.at<double>(2, 2)
	};
	ceres::Problem problem;

	std::map<int, double*> tr_opt, p2d, R_inv, t_inv;
	std::unordered_map<std::shared_ptr<Feature3D>, double*> p3d_opt;

	for (int i = fn - n; i < fn; i++)
	{
		Frame* frame = &tracker->frames[i];

		cv::Mat rod = cv::Mat_<double>(3, 1);
		cv::Mat R_transpose = cv::Mat_<double>(3, 3);
		cv::transpose(tracker->R[i], R_transpose);
		cv::Rodrigues(R_transpose, rod);

		tr_opt[i] = new double[6]{
			rod.at<double>(0), rod.at<double>(1), rod.at<double>(2),
			-tracker->t[i].at<double>(0), -tracker->t[i].at<double>(1), -tracker->t[i].at<double>(2)
		};

		if (i == 0)
			continue;

		for (auto& p : frame->map)
		{
			if (p.second.expired())
				continue;
			std::shared_ptr<Feature3D> f3d = p.second.lock();
			Feature f = p.first;

			cv::Point3f p3f = f3d->getPoint();

			p2d[i] = new double[2]{ (double)f.column, (double)f.row };

			if (!p3d_opt.count(f3d))
				p3d_opt[f3d] = new double[3]{ p3f.x, p3f.y, p3f.z };

			ceres::CostFunction* cost_function = ProjectionResidual::Create(p2d[i], _camera);
			problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), tr_opt[i], p3d_opt[f3d]);
		}
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 4;
	options.max_num_iterations = tracker->ba_iterations;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	if (tracker->verbose)
		std::cout << summary.FullReport() << "\n";

	// Updating 3D points and camera poses
	for (int i = fn - n; i < fn; i++)
	{
		if (i == 0)
			continue;

		double __t[] = { tr_opt[i][3] , tr_opt[i][4] , tr_opt[i][5] };
		double __rod[] = { tr_opt[i][0] , tr_opt[i][1] , tr_opt[i][2] };

		cv::Mat _R = cv::Mat_<double>(3, 3);
		cv::Mat _t = cv::Mat_<double>(3, 1, __t);
		cv::Mat rod = cv::Mat_<double>(3, 1, __rod);
		cv::Rodrigues(rod, _R);
		cv::transpose(_R, _R);

		tracker->R[i] = _R.clone();
		tracker->t[i] = -_t.clone();

		for (auto& p : p3d_opt)
		{
			p.first->update(p.second[0], p.second[1], p.second[2]);
		}
	}
}
