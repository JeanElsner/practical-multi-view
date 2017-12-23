#include "ProjectionResidual.h"

ceres::CostFunction* ProjectionResidual::Create(
	const double* p3d,
	const double* p2d,
	const double* camera) {
	return (new ceres::AutoDiffCostFunction<ProjectionResidual, 2, 9, 3>(
		new ProjectionResidual(p3d, p2d, camera)));
}
