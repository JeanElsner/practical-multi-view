#ifndef PROJECTION_RESIDUAL_H
#define PROJECTION_RESIDUAL_H

#include "Frame.h"
#include "Feature3D.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/calib3d.hpp>

class ProjectionResidual
{
public:

	const double* p2d;
	const double* camera;
	const double* R_inv;
	const double* t_inv;

	ProjectionResidual(
		const double* p2d, 
		const double* camera,
		const double* R_inv,
		const double* t_inv) : p2d(p2d), camera(camera), R_inv(R_inv), t_inv(t_inv) {}

	static ceres::CostFunction* ProjectionResidual::Create(
		const double* p2d,
		const double* camera,
		const double* R_inv,
		const double* t_inv);
	
	template <typename T>
	bool operator()(
		const T* const tr_opt,
		const T* const p3d_opt,
		T* residuals) const {
		
		/*T x_p, y_p, z_p;
		x_p = R[0] * p3d[0] + R[1] * p3d[1] + R[2] * p3d[2];
		y_p = R[3] * p3d[0] + R[4] * p3d[1] + R[5] * p3d[2];
		z_p = R[6] * p3d[0] + R[7] * p3d[1] + R[8] * p3d[2];*/

		T p3[3];
		p3[0] = p3d_opt[0] - t_inv[0];
		p3[1] = p3d_opt[1] - t_inv[1];
		p3[2] = p3d_opt[2] - t_inv[2];

		T p2[3];
		p2[0] = R_inv[0] * p3[0] + R_inv[1] * p3[1] + R_inv[2] * p3[2];
		p2[1] = R_inv[3] * p3[0] + R_inv[4] * p3[1] + R_inv[5] * p3[2];
		p2[2] = R_inv[6] * p3[0] + R_inv[7] * p3[1] + R_inv[8] * p3[2];

		p2[2] = p2[2]*(T)(-1);

		T p[3];
		ceres::AngleAxisRotatePoint(tr_opt, p2, p);
		
		p[0] += tr_opt[3];
		p[1] += tr_opt[4];
		p[2] += tr_opt[5];
		
		p[0] = p[0] / p[2] * camera[0] + camera[2];
		p[1] = p[1] / p[2] * camera[4] + camera[5];

		residuals[0] = (T)p2d[0] - p[0];
		residuals[1] = (T)p2d[1] - p[1];

		T test1 = (T)p2d[0] - p[0], test2 = (T)p2d[1] - p[1];

		return true;
	}
};

#endif