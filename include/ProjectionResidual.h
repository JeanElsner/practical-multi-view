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

	ProjectionResidual(
		const double* p2d, 
		const double* camera) : p2d(p2d), camera(camera) {}

	/**
		Creates a cost function based on this residual

		@param p2d A 2D feature
		@param camera Camera matrix
		@return The associated cost function
	**/
	static ceres::CostFunction* Create(
		const double* p2d,
		const double* camera);
	
	/**
		Operator overload as called by the ceres cost function

		@param tr_opt Camera pose that should be optimised
		@param p3d_opt 3D point that should be optimised
	**/
	template <typename T>
	bool operator()(
		const T* const tr_opt,
		const T* const p3d_opt,
		T* residuals) const {
		
		T p[3], p2[3];
		p2[0] = p3d_opt[0] + tr_opt[3];
		p2[1] = p3d_opt[1] + tr_opt[4];
		p2[2] = p3d_opt[2] + tr_opt[5];
		ceres::AngleAxisRotatePoint(tr_opt, p2, p);
		p[2] = p[2] * (T)(-1);
		
		p[0] = p[0] / p[2] * camera[0] + camera[2];
		p[1] = p[1] / p[2] * camera[4] + camera[5];

		residuals[0] = (T)p2d[0] - p[0];
		residuals[1] = (T)p2d[1] - p[1];

		return true;
	}
};

#endif