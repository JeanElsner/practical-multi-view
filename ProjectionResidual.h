#ifndef PROJECTION_RESIDUAL_H
#define PROJECTION_RESIDUAL_H

#include "Frame.h"
#include "Feature3D.h"
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>

class ProjectionResidual
{
public:

	const double* p3d;
	const double* p2d;
	const double* camera;

	ProjectionResidual(
		const double* p3d, 
		const double* p2d, 
		const double* camera) : p3d(p3d), p2d(p2d), camera(camera) {}

	static ceres::CostFunction* ProjectionResidual::Create(
		const double* p3d,
		const double* p2d,
		const double* camera);
	
	template <typename T>
	bool operator()(
		const T* const R,
		const T* const t,
		T* residuals) const {
		
		T x_p, y_p, z_p;
		x_p = R[0] * p3d[0] + R[1] * p3d[1] + R[2] * p3d[2];
		y_p = R[3] * p3d[0] + R[4] * p3d[1] + R[5] * p3d[2];
		z_p = R[6] * p3d[0] + R[7] * p3d[1] + R[8] * p3d[2];
		
		x_p += t[0];
		y_p += t[1];
		z_p += t[2];
		
		x_p = x_p / z_p * camera[0] + camera[2];
		y_p = y_p / z_p * camera[4] + camera[5];

		residuals[0] = (T)p2d[0] - x_p;
		residuals[1] = (T)p2d[1] - y_p;

		return true;
	}
};

#endif