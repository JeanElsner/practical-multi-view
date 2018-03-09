#ifndef BASE_PNP_SOLVER_H
#define BASE_PNP_SOLVER_H
#include "Frame.h"

// Abstract base class, used to estimate camera poses from 2D-3D correspondences
class BasePnPSolver
{
public:

	/**
		Solves the point-n-perspective problem and estimates a camera pose.
		Additionally, may remove 3D point outliers.

		@param src First image
		@param next Second image
		@param R Estimated rotation
		@param t Estimated translation
	*/
	virtual void solvePnP(Frame& src, Frame& next, cv::Mat& R, cv::Mat& t) = 0;
};
#endif
