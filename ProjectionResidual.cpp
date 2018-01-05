#include "ProjectionResidual.h"

ceres::CostFunction* ProjectionResidual::Create(
	const double* p2d,
	const double* camera,
	const double* R_inv,
	const double* t_inv) {
	return (new ceres::AutoDiffCostFunction<ProjectionResidual, 2, 6, 3>(
		new ProjectionResidual(p2d, camera, R_inv, t_inv)));
}
